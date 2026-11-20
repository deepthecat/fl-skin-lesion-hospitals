"""
huan_luyen_dp.py — RQ2 phần 2: differential privacy (DP-SGD) bên trong từng máy trạm.

    python huan_luyen_dp.py --kich-ban iid --epsilon 8
    python huan_luyen_dp.py --kich-ban mot_may --epsilon 3      # "tập trung có DP"
    python huan_luyen_dp.py --kich-ban iid --epsilon inf         # cùng mô hình, KHÔNG DP -> mốc so sánh

Ý tưởng DP-SGD (Abadi và c.s., 2016), kể bằng lời:
  Mỗi bước huấn luyện, tính gradient RIÊNG cho từng ảnh, rồi (1) CẮT mỗi gradient về độ dài tối đa C, để
  không ảnh nào kéo mô hình mạnh hơn C; (2) cộng chúng lại và thêm NHIỄU Gauss cỡ σ·C. Nhờ vậy, có hay
  không có một bệnh nhân cụ thể trong dữ liệu thì mô hình ra gần như nhau — sai khác bị chặn bởi ε.
  ε nhỏ = kín hơn = nhiễu nhiều hơn = mô hình kém hơn. Đó là sự đánh đổi RQ2 cần đo.

Trong federated: mỗi bệnh viện tự chạy DP-SGD trên dữ liệu của mình. Mô hình toàn cục là trung bình các
mô hình đã-có-DP, nên kế thừa bảo đảm đó (hậu xử lý không làm mất DP). Bảo đảm là cho TỪNG ẢNH trong
dữ liệu của mỗi bệnh viện, với ε tính trên toàn bộ các vòng.
Không dùng kiểu DP mức-máy-trạm (thêm nhiễu ở máy chủ) vì nó cần hàng nghìn máy trạm mới có ε dùng được;
ta chỉ có 5.

Ba ràng buộc kỹ thuật, ghi rõ vì chúng ảnh hưởng tới cách đọc kết quả:
  - BatchNorm không tương thích DP-SGD (nó trộn thông tin giữa các ảnh trong lô). Thư viện Opacus thay nó
    bằng GroupNorm. Trọng số GroupNorm khởi tạo mới, nên mô hình này KHÁC mô hình ở các thí nghiệm trước.
    Vì vậy mốc so sánh là cùng mô hình GroupNorm chạy KHÔNG DP (--epsilon inf), không phải kết quả cũ.
  - Gradient riêng từng ảnh tốn bộ nhớ gấp nhiều lần: mặc định ResNet18 ở 128 px, lô vật lý 16 ảnh.
    Kênh của ResNet18 (64/128/256/512) chia hết cho 32 nhóm GroupNorm; MobileNetV3 thì không.
  - Không dùng AMP (half precision) ở mọi nhánh của script này, kể cả nhánh không DP, để so cho công bằng.

Mọi nhánh trong script (kể cả inf) dùng chung: kiến trúc GroupNorm, 128 px, AdamW lr 5e-4, 30 vòng,
cùng tăng cường và trọng số lớp. Khác biệt duy nhất giữa các nhánh là ε.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from du_lieu import RA, cham, doc_chia_tap, ma_file, nap_dem, tao_mo_hinh
from huan_luyen_lien_ket import gop, luu_an_toan
from huan_luyen_tap_trung import lo, suy_luan, trong_so_lop


class ChiSo(Dataset):
    """Trả về chỉ số ảnh; ảnh thật lấy từ bộ đệm trên GPU trong vòng lặp (nhanh hơn DataLoader nạp ảnh)."""
    def __init__(self, idx):
        self.idx = np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return (torch.tensor(int(self.idx[i])),)      # Opacus đòi mỗi mẫu là một bộ


def _basic_block_khong_tai_cho(self, x):
    """Bản sao BasicBlock.forward của timm, chỉ đổi `x += shortcut` thành `x = x + shortcut`:
    phép cộng tại chỗ làm hỏng móc gradient từng ảnh của Opacus."""
    shortcut = x
    x = self.act1(self.drop_block(self.bn1(self.conv1(x))))
    x = self.bn2(self.conv2(self.aa(x)))
    if self.se is not None:
        x = self.se(x)
    if self.drop_path is not None:
        x = self.drop_path(x)
    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    return self.act2(x + shortcut)


def _va_resnet(mh):
    import timm.models.resnet as r
    can = ("conv1", "bn1", "drop_block", "act1", "aa", "conv2", "bn2", "se", "drop_path", "downsample", "act2")
    for m in mh.modules():
        if type(m) is r.BasicBlock:
            thieu = [a for a in can if not hasattr(m, a)]
            if thieu:
                raise SystemExit(f"Phiên bản timm khác dự kiến (BasicBlock thiếu {thieu}) — báo lại để sửa.")
            m.forward = _basic_block_khong_tai_cho.__get__(m)
        elif type(m) is r.Bottleneck:
            raise SystemExit("Chỉ vá sẵn cho ResNet18/34 (BasicBlock). Dùng --backbone resnet18.")


def tao_mo_hinh_dp(ten, pretrained):
    from opacus.validators import ModuleValidator
    mh = ModuleValidator.fix(tao_mo_hinh(ten, pretrained=pretrained))
    for m in mh.modules():                  # ReLU "sửa tại chỗ" đụng với móc gradient từng ảnh của Opacus
        if hasattr(m, "inplace"):
            m.inplace = False
    if ten.startswith("resnet"):
        _va_resnet(mh)
    loi = ModuleValidator.validate(mh, strict=False)
    if loi:
        raise SystemExit(f"Mô hình chưa tương thích DP-SGD: {loi[:3]}")
    return mh


def tinh_sigma(eps, delta, q, buoc):
    from opacus.accountants.utils import get_noise_multiplier
    return float(get_noise_multiplier(target_epsilon=eps, target_delta=delta, sample_rate=q, steps=buoc,
                                      accountant="rdp"))


def tinh_eps(sigma, q, buoc, delta):
    from opacus.accountants import RDPAccountant
    a = RDPAccountant()
    a.history = [(sigma, q, buoc)]
    return float(a.get_epsilon(delta))


def cuc_bo_khong_dp(mh, w0, X, y, idx, arg, lr, w_lop, tb, rng):
    mh.load_state_dict(w0)
    mh.train()
    opt = torch.optim.AdamW(mh.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(arg.epoch_cuc_bo):
        p = rng.permutation(idx)
        for i in range(0, len(p), arg.bs):
            b = p[i:i + arg.bs]
            if len(b) < 2:
                continue
            loss = F.cross_entropy(mh(lo(X, b, tb, tang=True)), torch.from_numpy(y[b]).to(tb), weight=w_lop)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in mh.state_dict().items()}


def cuc_bo_dp(gs, w0, X, y, idx, arg, lr, w_lop, tb, sigma, seed):
    from opacus.data_loader import DPDataLoader
    from opacus.optimizers import DPOptimizer
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    gs._module.load_state_dict(w0)
    gs.train()
    g = torch.Generator().manual_seed(seed)
    dl = DPDataLoader.from_data_loader(DataLoader(ChiSo(idx), batch_size=arg.bs), generator=g)
    opt = DPOptimizer(torch.optim.AdamW(gs.parameters(), lr=lr, weight_decay=1e-4), noise_multiplier=sigma,
                      max_grad_norm=arg.clip, expected_batch_size=arg.bs)
    for _ in range(arg.epoch_cuc_bo):
        with BatchMemoryManager(data_loader=dl, max_physical_batch_size=arg.bs_vat_ly, optimizer=opt) as mdl:
            for (b,) in mdl:
                b = b.numpy().reshape(-1)
                if len(b) == 0:                  # lấy mẫu Poisson thỉnh thoảng ra lô rỗng
                    continue
                loss = F.cross_entropy(gs(lo(X, b, tb, tang=True)), torch.from_numpy(y[b]).to(tb), weight=w_lop)
                opt.zero_grad()
                loss.backward()
                opt.step()
    return {k: v.detach().cpu().clone() for k, v in gs._module.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kich-ban", default="iid", help="iid, dir1.0, dir0.3, dir0.1 hoặc mot_may")
    ap.add_argument("--epsilon", default="8", help="ngân sách riêng tư, hoặc 'inf' = không DP")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0, help="độ dài tối đa gradient mỗi ảnh (C)")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--kt", type=int, default=128)
    ap.add_argument("--vong", type=int, default=30)
    ap.add_argument("--epoch-cuc-bo", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64, help="lô logic (kỳ vọng) của DP-SGD")
    ap.add_argument("--bs-vat-ly", type=int, default=16, help="lô thực sự đưa vào GPU; nhỏ để vừa 4 GB")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--trong-so", default="nghich", choices=("nghich", "can", "khong"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--khong-pretrained", action="store_true")
    ap.add_argument("--dung-sau-vong", type=int, default=0)
    ap.add_argument("--ra", default=str(RA / "dp"))
    arg = ap.parse_args()
    arg.thuat_toan = "fedavg"                   # gop() cần thuộc tính này
    eps = math.inf if arg.epsilon.lower() == "inf" else float(arg.epsilon)

    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    X = nap_dem(ds, arg.kt)
    y = ds.nhan.values
    I = {p: np.flatnonzero(ds.phan.values == p) for p in ("train", "val", "val_chinh_thuc", "test")}
    if "trung_train" in ds.columns:
        I["test_sach"] = np.flatnonzero((ds.phan.values == "test") & (ds.trung_train.values == 0))
    if arg.kich_ban == "mot_may":
        may = {0: I["train"]}
    else:
        m = ds[f"may_{arg.kich_ban}"].values
        may = {int(k): np.flatnonzero((ds.phan.values == "train") & (m == k)) for k in sorted(np.unique(m[I["train"]]))}

    # ---- ngân sách riêng tư: tính σ cho từng máy trạm để sau arg.vong vòng, mỗi máy trạm tiêu đúng ε
    thong_tin = {}
    for k, idx in may.items():
        buoc_moi_epoch = math.ceil(len(idx) / arg.bs)
        q = 1.0 / buoc_moi_epoch
        buoc = arg.vong * arg.epoch_cuc_bo * buoc_moi_epoch
        sigma = 0.0 if math.isinf(eps) else tinh_sigma(eps, arg.delta, q, buoc)
        thong_tin[k] = dict(n=len(idx), q=q, buoc=buoc, sigma=sigma,
                            eps_that=(math.inf if sigma == 0 else tinh_eps(sigma, q, buoc, arg.delta)))
    print(f"DP ε={arg.epsilon} δ={arg.delta} C={arg.clip} | {arg.kich_ban} | {arg.backbone} {arg.kt} | {tb}")
    for k, t in thong_tin.items():
        print(f"  máy trạm {k}: {t['n']} ảnh, q={t['q']:.4f}, {t['buoc']} bước, σ={t['sigma']:.3f}, ε thực={t['eps_that']:.2f}")

    ten = f"dp_{arg.kich_ban}_eps{arg.epsilon}_{arg.backbone}_kt{arg.kt}_s{arg.seed}"
    thu_muc = Path(arg.ra); thu_muc.mkdir(parents=True, exist_ok=True)
    f_ckpt, f_json = thu_muc / f"{ten}_ckpt.pt", thu_muc / f"{ten}.json"
    if f_json.exists():
        print(f"Đã có {f_json.name} — bỏ qua.")
        return

    torch.manual_seed(arg.seed)
    mh = tao_mo_hinh_dp(arg.backbone, pretrained=not arg.khong_pretrained).to(tb)
    gs = None
    if not math.isinf(eps):
        from opacus import GradSampleModule
        gs = GradSampleModule(mh)
    w_lop = trong_so_lop(y[I["train"]], arg.trong_so)
    w_lop = w_lop.to(tb) if w_lop is not None else None
    cau_hinh = {k: v for k, v in vars(arg).items() if k != "dung_sau_vong"}

    if f_ckpt.exists():
        ck = torch.load(f_ckpt, map_location="cpu", weights_only=False)
        if ck["cau_hinh"] != cau_hinh:
            raise SystemExit("Checkpoint có cấu hình khác lệnh hiện tại.")
        w_tc, vong0, tot, w_tot, lich_su, giay0 = ck["w"], ck["vong"], ck["tot"], ck["w_tot"], ck["lich_su"], ck["giay"]
        rng = np.random.default_rng(); rng.bit_generator.state = ck["rng"]
        print(f"Chạy tiếp từ vòng {vong0 + 1}/{arg.vong}")
    else:
        w_tc = {k: v.detach().cpu().clone() for k, v in mh.state_dict().items()}
        vong0, tot, w_tot, lich_su, giay0 = 0, -1.0, None, [], 0.0
        rng = np.random.default_rng(arg.seed)

    t0 = time.time() - giay0
    for vong in range(vong0, arg.vong):
        lr = arg.lr * 0.5 * (1 + math.cos(math.pi * vong / arg.vong))
        cac_w, so_anh = [], []
        for k, idx in may.items():
            if gs is None:
                w_k = cuc_bo_khong_dp(mh, w_tc, X, y, idx, arg, lr, w_lop, tb, rng)
            else:
                w_k = cuc_bo_dp(gs, w_tc, X, y, idx, arg, lr, w_lop, tb, thong_tin[k]["sigma"],
                                seed=arg.seed * 100000 + vong * 100 + k)
            cac_w.append(w_k); so_anh.append(len(idx))
        w_tc = gop(w_tc, cac_w, so_anh, arg, {})
        del cac_w
        mh.load_state_dict(w_tc)
        v = cham(y[I["val"]], suy_luan(mh, X, I["val"], tb))
        dau = ""
        if v["bacc"] > tot:
            tot, w_tot, dau = v["bacc"], copy.deepcopy(w_tc), " *"
        lich_su.append(dict(vong=vong + 1, lr=lr, **{f"val_{k}": x for k, x in v.items()}))
        print(f"vòng {vong+1:>3}/{arg.vong}  val bacc {v['bacc']:.4f}  f1 {v['f1_macro']:.4f}  "
              f"acc {v['acc']:.4f}  ({time.time()-t0:.0f}s){dau}", flush=True)
        luu_an_toan(dict(cau_hinh=cau_hinh, w=w_tc, vong=vong + 1, tot=tot, w_tot=w_tot, lich_su=lich_su,
                         giay=time.time() - t0, rng=rng.bit_generator.state), f_ckpt)
        if arg.dung_sau_vong and vong + 1 - vong0 >= arg.dung_sau_vong:
            print("(dừng thử theo --dung-sau-vong)")
            return

    # Lưu ý: chọn vòng tốt nhất theo val là một lần "nhìn dữ liệu" nữa. Val không thuộc máy trạm nào nên
    # không tốn ngân sách ε của máy trạm, nhưng phải ghi rõ trong bài.
    mh.load_state_dict(w_tot)
    torch.save(w_tot, thu_muc / f"{ten}.pt")
    logit = suy_luan(mh, X, np.arange(len(ds)), tb)
    ket = {p: cham(y[i], logit[i]) for p, i in I.items() if len(i)}
    np.savez_compressed(thu_muc / f"{ten}_logit.npz", image=np.asarray(ds.image.tolist(), dtype=np.str_),
                        phan=np.asarray(ds.phan.tolist(), dtype=np.str_), nhan=y, logit=logit.astype(np.float32))
    json.dump(dict(ten=ten, ma_script=ma_file(__file__), ma_du_lieu=ma_file(Path(__file__).parent / "du_lieu.py"),
                   cau_hinh=cau_hinh, epsilon=arg.epsilon, may_tram={str(k): t for k, t in thong_tin.items()},
                   giay=time.time() - t0, ket_qua=ket, lich_su=lich_su),
              open(f_json, "w"), indent=1, ensure_ascii=False, default=str)
    f_ckpt.unlink(missing_ok=True)
    print("\nKẾT QUẢ (mô hình toàn cục tốt nhất trên val):")
    for p, d in ket.items():
        print(f"  {p:<15} bacc {d['bacc']:.4f}  f1 {d['f1_macro']:.4f}  acc {d['acc']:.4f}  auc {d['auc_macro']:.4f}")
    print(f"Đã ghi {f_json}")


if __name__ == "__main__":
    main()
