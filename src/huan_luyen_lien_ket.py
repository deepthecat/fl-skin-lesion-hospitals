"""
huan_luyen_lien_ket.py — học liên kết (federated) mô phỏng trên một máy.

    python huan_luyen_lien_ket.py --thuat-toan fedavg --kich-ban dir0.3 --backbone efficientnet_b0
    python huan_luyen_lien_ket.py --thuat-toan fedavg --kich-ban mot_may     # kiểm tra tính đúng (xem dưới)

Mỗi VÒNG:
  1. Máy chủ gửi mô hình toàn cục cho cả K máy trạm (cross-silo: bệnh viện nào cũng tham gia mọi vòng).
  2. Mỗi máy trạm huấn luyện trên dữ liệu CỦA RIÊNG NÓ vài epoch. Dữ liệu không rời máy trạm.
  3. Máy trạm gửi lại trọng số. Máy chủ gộp lại theo thuật toán, được mô hình toàn cục mới.
  4. Chấm mô hình toàn cục trên validation (tách theo tổn thương, không thuộc máy trạm nào).

Ba thuật toán:
  fedavg  — trung bình có trọng số theo số ảnh mỗi máy trạm (McMahan và c.s., 2017).
  fedprox — như fedavg, nhưng mỗi máy trạm bị phạt nếu trọng số đi quá xa mô hình toàn cục:
            loss += (mu/2)·||w - w_toan_cuc||². Chống "trôi" khi dữ liệu các máy trạm lệch nhau
            (Li và c.s., 2020).
  fedavgm — như fedavg, nhưng máy chủ cập nhật có momentum, làm mượt các bước gộp khi máy trạm
            kéo mô hình về các hướng khác nhau (Hsu và c.s., 2019 — cũng là bài đề xuất cách chia
            Dirichlet mà ta dùng).
  fedbn   — như fedavg, nhưng BatchNorm (trọng số + thống kê chạy) GIỮ RIÊNG ở từng máy trạm, không gửi
            lên gộp (Li và c.s., 2021). Nhắm đúng giả thuyết G8b: khác biệt máy chụp hại FL qua thống kê
            chuẩn hoá. Đánh giá: ảnh của bệnh viện k chấm bằng mô hình của k (val, kịch bản nguồn thật —
            "theo nguồn"); ảnh không rõ nguồn (test) chấm bằng TRUNG BÌNH xác suất của K mô hình ("ensemble").
            Mỗi vòng ghi bn_kc = khoảng cách tương đối giữa thống kê BN của các máy trạm.

Để so công bằng với baseline tập trung (RQ1), mọi thứ khác GIỮ NGUYÊN: cùng kiến trúc, cùng tăng
cường, cùng AdamW, cùng trọng số lớp. Khoảng cách còn lại là do federated, không do cấu hình khác.

Trọng số lớp tính trên TOÀN BỘ tập train — tức là giả định máy chủ biết phân bố nhãn chung. Đây là
giả định phổ biến và phải ghi rõ trong bài.

KIỂM TRA TÍNH ĐÚNG: --kich-ban mot_may dồn hết dữ liệu vào một máy trạm. Khi đó fedavg về lý thuyết
tương đương huấn luyện tập trung. Nếu kết quả lệch xa baseline tập trung -> code federated có lỗi.

Chạy lại được sau khi bị ngắt: lưu checkpoint sau MỖI vòng. Gõ lại đúng lệnh cũ là nó chạy tiếp.
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

from nhat_ky import NhatKy
from du_lieu import RA, TIEN_XU_LY, cham, doc_chia_tap, ma_file, nap_dem, tao_mo_hinh
from huan_luyen_tap_trung import lo, suy_luan, trong_so_lop


# Mức đóng băng: tiền tố tên các module VẪN ĐƯỢC HUẤN LUYỆN. Mọi thứ khác đóng băng ở trọng số ImageNet.
MO_KHOA = {
    "efficientnet_b0": {"dau": ["classifier"],
                        "k1": ["conv_head", "bn2", "classifier"],
                        "k2": ["blocks.6", "conv_head", "bn2", "classifier"]},
    "resnet18": {"dau": ["fc"], "k1": ["layer4.1", "fc"], "k2": ["layer4", "fc"]},
}


def dong_bang(mh, ten, muc):
    """Đóng băng theo mức. Trả về danh sách BatchNorm bị đóng băng — chúng phải ở chế độ suy luận khi
    huấn luyện, nếu không thống kê chạy (running mean/var) vẫn trôi theo dữ liệu từng bệnh viện."""
    if muc == "khong":
        return []
    if ten not in MO_KHOA:
        raise SystemExit(f"Chưa định nghĩa mức đóng băng cho {ten}")
    mo = MO_KHOA[ten][muc]
    for n, p in mh.named_parameters():
        p.requires_grad = any(n == t or n.startswith(t + ".") for t in mo)
    bn = []
    for n, m in mh.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not any(n == t or n.startswith(t + ".") for t in mo):
            bn.append(m)
    return bn


def la_float(t):
    return torch.is_floating_point(t)


def khoa_bn(mh):
    """Các khoá state_dict thuộc BatchNorm: weight, bias, running_mean, running_var, num_batches_tracked."""
    ten = [n for n, m in mh.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    return {k for k in mh.state_dict() if any(k.startswith(n + ".") for n in ten)}


def khoang_cach_bn(w_bn):
    """G8b: trung bình qua các lớp BN của ||μ_i − μ_j|| / (trung bình ||μ||), cho mọi cặp máy trạm."""
    ks = sorted(k for k in w_bn[0] if k.endswith("running_mean"))
    K = len(w_bn)
    D = np.zeros((K, K))
    for k in ks:
        mu = [w[k].float() for w in w_bn]
        for i in range(K):
            for j in range(i + 1, K):
                d = float((mu[i] - mu[j]).norm() / (0.5 * (mu[i].norm() + mu[j].norm()) + 1e-8))
                D[i, j] += d / len(ks); D[j, i] = D[i, j]
    return dict(bn_kc=[[round(x, 4) for x in r] for r in D.tolist()],
                bn_kc_tb=round(float(D[np.triu_indices(K, 1)].mean()), 4))


def du_doan_fedbn(mh, w_tc, w_bn, X, idx, tb):
    """Trả về (logit ensemble, list logit của từng máy trạm) cho các ảnh idx."""
    rieng = []
    for wb in w_bn:
        mh.load_state_dict({**w_tc, **wb})
        rieng.append(suy_luan(mh, X, idx, tb))
    p = np.mean([np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True)
                 for l in rieng], axis=0)
    return np.log(np.clip(p, 1e-12, 1)), rieng


def logit_theo_nguon(rieng, ens, may_cua):
    """Ảnh có máy trạm k (may_cua ≥ 0) lấy logit của mô hình k; ảnh không rõ (−1) lấy ensemble."""
    ra = ens.copy()
    for k, l in enumerate(rieng):
        m = may_cua == k
        ra[m] = l[m]
    return ra


def huan_luyen_may_tram(mh, w0, X, y, idx, arg, lr, w_lop, tb, rng, scaler, bn_dong=()):
    """Huấn luyện cục bộ, trả về state_dict (trên CPU)."""
    mh.load_state_dict(w0)
    mh.train()
    for m in bn_dong:
        m.eval()
    mo = [p for p in mh.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(mo, lr=lr, weight_decay=1e-4)
    goc = [p.detach().clone() for p in mo] if arg.thuat_toan == "fedprox" else None
    so_buoc = 0
    for _ in range(arg.epoch_cuc_bo):
        p = rng.permutation(idx)
        for i in range(0, len(p), arg.bs):
            b = p[i:i + arg.bs]
            if len(b) < 2:                      # BatchNorm không chạy với lô 1 ảnh
                continue
            xb = lo(X, b, tb, tang=True)
            yb = torch.from_numpy(y[b]).to(tb)
            with torch.autocast(tb.type, enabled=tb.type == "cuda"):
                loss = F.cross_entropy(mh(xb), yb, weight=w_lop)
            if goc is not None:
                loss = loss + (arg.mu / 2) * sum(((q - g) ** 2).sum() for q, g in zip(mo, goc))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            so_buoc += 1
    return {k: v.detach().cpu().clone() for k, v in mh.state_dict().items()}, so_buoc


def gop(w_toan_cuc, cac_w, trong_so, arg, dong_luong):
    """Gộp trọng số các máy trạm. Tham số float: trung bình có trọng số (+ momentum nếu fedavgm).
    Bộ đếm nguyên của BatchNorm: lấy của máy trạm đầu (không có nghĩa khi lấy trung bình)."""
    tong = sum(trong_so)
    moi = {}
    for k, v0 in w_toan_cuc.items():
        if not la_float(v0):
            moi[k] = cac_w[0][k].clone()
            continue
        tb_ = sum(w[k].float() * (n / tong) for w, n in zip(cac_w, trong_so))
        if arg.thuat_toan == "fedavgm" and not k.endswith(("running_mean", "running_var")):
            delta = v0.float() - tb_                                  # "gradient giả" của máy chủ
            dong_luong[k] = arg.beta * dong_luong.get(k, torch.zeros_like(delta)) + delta
            tb_ = v0.float() - arg.lr_may_chu * dong_luong[k]
        moi[k] = tb_.to(v0.dtype)
    return moi


def do_huong(w_tc, cac_w, so_anh):
    """G8/G9: mỗi máy trạm kéo mô hình chung đi bao xa (do_troi = độ dài cập nhật) và theo hướng nào.
    cos[i][j] = cosine giữa cập nhật của máy i và máy j: gần 1 = kéo cùng chiều, gần 0/âm = kéo lệch nhau.
    cos_tb[i] = cosine giữa cập nhật của máy i và cập nhật trung bình (cái máy chủ thực sự áp dụng)."""
    k_ = [k for k, v in w_tc.items() if la_float(v) and not k.endswith(("running_mean", "running_var"))]
    D = torch.stack([torch.cat([(w[k].float() - w_tc[k].float()).flatten() for k in k_]) for w in cac_w])
    tb_ = (D * torch.tensor(so_anh, dtype=torch.float32)[:, None]).sum(0) / sum(so_anh)
    chuan = D.norm(dim=1)
    Dn = D / chuan.clamp_min(1e-12)[:, None]
    return dict(do_troi=[round(x, 5) for x in chuan.tolist()],
                cos=[[round(x, 4) for x in r] for r in (Dn @ Dn.T).tolist()],
                cos_tb=[round(x, 4) for x in (Dn @ (tb_ / tb_.norm().clamp_min(1e-12))).tolist()])


def luu_an_toan(obj, f):
    tam = f.with_suffix(".tmp")
    torch.save(obj, tam)
    tam.replace(f)                              # đổi tên là thao tác nguyên tử: không bao giờ để lại file hỏng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thuat-toan", default="fedavg", choices=("fedavg", "fedprox", "fedavgm", "fedbn"))
    ap.add_argument("--kich-ban", default="iid",
                    help="nguon (4 bệnh viện thật), iid, dir1.0, dir0.3, dir0.1, iid_anh (cột may_* trong chia_tap.csv) hoặc mot_may")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--kt", type=int, default=224)
    ap.add_argument("--vong", type=int, default=30)
    ap.add_argument("--epoch-cuc-bo", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mu", type=float, default=0.01, help="hệ số phạt của fedprox")
    ap.add_argument("--beta", type=float, default=0.9, help="momentum máy chủ của fedavgm")
    ap.add_argument("--lr-may-chu", type=float, default=1.0)
    ap.add_argument("--trong-so", default="nghich", choices=("nghich", "can", "khong"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tien-xu-ly", default="p0", choices=tuple(TIEN_XU_LY),
                    help="; ".join(f"{k}: {v}" for k, v in TIEN_XU_LY.items()))
    ap.add_argument("--trong-so-cuc-bo", action="store_true",
                    help="mỗi máy trạm tự tính trọng số lớp từ dữ liệu của mình (không ai khai số ca cho máy chủ)")
    ap.add_argument("--dong-bang", default="khong", choices=("khong", "dau", "k1", "k2"),
                    help="khong = huấn luyện toàn mạng; dau = chỉ tầng phân loại; k1, k2 = thêm dần các khối cuối")
    ap.add_argument("--khong-pretrained", action="store_true", help="chỉ để thử code")
    ap.add_argument("--ghi-huong", action="store_true",
                    help="mỗi vòng ghi độ trôi (độ dài cập nhật) và cosine giữa cập nhật các máy trạm; thêm _gh vào tên")
    ap.add_argument("--dung-sau-vong", type=int, default=0, help="chỉ để thử khả năng chạy tiếp: tự dừng sau N vòng")
    ap.add_argument("--ra", default=str(RA / "lien_ket"))
    arg = ap.parse_args()

    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    X = nap_dem(ds, arg.kt, arg.tien_xu_ly)
    y = ds.nhan.values
    I = {p: np.flatnonzero(ds.phan.values == p) for p in ("train", "val", "val_chinh_thuc", "test")}
    if "trung_train" in ds.columns:
        I["test_sach"] = np.flatnonzero((ds.phan.values == "test") & (ds.trung_train.values == 0))

    # ---- máy trạm
    if arg.kich_ban == "mot_may":
        may = {0: I["train"]}
    else:
        cot = f"may_{arg.kich_ban}"
        if cot not in ds.columns:
            raise SystemExit(f"Không có cột {cot} trong chia_tap.csv")
        m = ds[cot].values
        may = {int(k): np.flatnonzero((ds.phan.values == "train") & (m == k))
               for k in sorted(np.unique(m[I["train"]]))}
    n_may = {k: len(v) for k, v in may.items()}
    may_cua = np.full(len(ds), -1)                     # máy trạm "chủ" của từng ảnh, −1 = không rõ (dùng ensemble)
    for k, idx in may.items():
        may_cua[idx] = k
    if arg.kich_ban == "nguon" and "nguon" in ds.columns:
        nguon_ve_may = {ds.nguon.values[idx[0]]: k for k, idx in may.items()}
        for i in I["val"]:
            may_cua[i] = nguon_ve_may.get(ds.nguon.values[i], -1)
    print(f"{arg.thuat_toan} | {arg.kich_ban} | {arg.backbone} {arg.kt} | {len(may)} máy trạm: {n_may} | {tb}")

    db = "" if arg.dong_bang == "khong" else f"_db{arg.dong_bang}"
    txl = "" if arg.tien_xu_ly == "p0" else f"_txl{arg.tien_xu_ly}"     # p0 giữ tên cũ
    cb = "_cb" if arg.trong_so_cuc_bo else ""
    mu = f"_mu{arg.mu:g}" if arg.thuat_toan == "fedprox" and arg.mu != 0.01 else ""   # μ mặc định giữ tên cũ
    gh = "_gh" if arg.ghi_huong else ""
    vg = f"_v{arg.vong}" if arg.vong != 30 else ""   # số vòng khác mặc định -> tên riêng, không đè run cũ trên wandb
    hau_ab = ("_khp" if arg.khong_pretrained else "") + ("" if arg.trong_so == "nghich" else f"_ts{arg.trong_so}") \
        + ("" if arg.epoch_cuc_bo == 1 else f"_e{arg.epoch_cuc_bo}")      # ablation: tên riêng, không đè run gốc
    ten = f"{arg.thuat_toan}{mu}_{arg.kich_ban}_{arg.backbone}{db}{txl}{cb}{gh}{vg}{hau_ab}_kt{arg.kt}_s{arg.seed}"
    thu_muc = Path(arg.ra); thu_muc.mkdir(parents=True, exist_ok=True)
    f_ckpt, f_json = thu_muc / f"{ten}_ckpt.pt", thu_muc / f"{ten}.json"
    if f_json.exists():
        print(f"Đã có {f_json.name} — bỏ qua (xoá file đó nếu muốn chạy lại).")
        return

    torch.manual_seed(arg.seed)
    mh = tao_mo_hinh(arg.backbone, pretrained=not arg.khong_pretrained).to(tb)
    bn_dong = dong_bang(mh, arg.backbone, arg.dong_bang)
    so_mo = sum(p.numel() for p in mh.parameters() if p.requires_grad)
    print(f"Tham số được huấn luyện: {so_mo:,} / {sum(p.numel() for p in mh.parameters()):,}"
          f"  (đóng băng: {arg.dong_bang}, {len(bn_dong)} BatchNorm giữ cố định)")
    # trọng số lớp: toàn cục (máy chủ cần biết số ca từng bệnh của mọi máy trạm) hoặc cục bộ (mỗi máy tự tính)
    if arg.trong_so_cuc_bo:
        w_may = {k: trong_so_lop(y[idx], arg.trong_so) for k, idx in may.items()}
    else:
        w_g = trong_so_lop(y[I["train"]], arg.trong_so)
        w_may = {k: w_g for k in may}
    w_may = {k: (w.to(tb) if w is not None else None) for k, w in w_may.items()}
    scaler = torch.amp.GradScaler(enabled=tb.type == "cuda")

    # ---- trạng thái (hoặc nạp lại nếu đã chạy dở)
    if f_ckpt.exists():
        ck = torch.load(f_ckpt, map_location="cpu", weights_only=False)
        if ck["cau_hinh"] != {k: v for k, v in vars(arg).items() if k != "dung_sau_vong"}:
            raise SystemExit("Checkpoint có cấu hình khác lệnh hiện tại — xoá checkpoint hoặc dùng đúng lệnh cũ.")
        w_tc, vong0, tot, w_tot = ck["w"], ck["vong"], ck["tot"], ck["w_tot"]
        w_bn, w_bn_tot = ck.get("w_bn"), ck.get("w_bn_tot")
        lich_su, dong_luong, giay0 = ck["lich_su"], ck["dong_luong"], ck["giay"]
        rng = np.random.default_rng(); rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"Chạy tiếp từ vòng {vong0 + 1}/{arg.vong} (tốt nhất hiện tại: val bacc {tot:.4f})")
    else:
        w_tc = {k: v.detach().cpu().clone() for k, v in mh.state_dict().items()}
        vong0, tot, w_tot, lich_su, dong_luong, giay0 = 0, -1.0, None, [], {}, 0.0
        rng = np.random.default_rng(arg.seed)
        w_bn, w_bn_tot = None, None
    KBN = khoa_bn(mh)
    if arg.thuat_toan == "fedbn" and w_bn is None:
        w_bn = [{k: w_tc[k].clone() for k in KBN} for _ in may]
        print(f"FedBN: {len(KBN)} khoá BatchNorm giữ riêng ở mỗi máy trạm")

    t0 = time.time() - giay0
    nk = NhatKy(ten, {k: v for k, v in vars(arg).items() if k != "dung_sau_vong"}, "lien_ket",
                [__file__, Path(__file__).parent / "du_lieu.py"],
                them=dict(may_tram=str(n_may), so_tham_so_mo=so_mo, ma_script=ma_file(__file__)))
    for vong in range(vong0, arg.vong):
        lr = arg.lr * 0.5 * (1 + math.cos(math.pi * vong / arg.vong))     # cosine theo vòng
        cac_w, so_anh, buoc = [], [], []
        for k, idx in may.items():
            w0 = {**w_tc, **w_bn[k]} if arg.thuat_toan == "fedbn" else w_tc
            w_k, s = huan_luyen_may_tram(mh, w0, X, y, idx, arg, lr, w_may[k], tb, rng, scaler, bn_dong)
            if arg.thuat_toan == "fedbn":
                w_bn[k] = {kk: w_k[kk].clone() for kk in KBN}
            cac_w.append(w_k); so_anh.append(len(idx)); buoc.append(s)
        huong = do_huong(w_tc, cac_w, so_anh) if arg.ghi_huong else {}
        w_tc = gop(w_tc, cac_w, so_anh, arg, dong_luong)
        del cac_w

        them = {}
        if arg.thuat_toan == "fedbn":
            ens, rieng = du_doan_fedbn(mh, w_tc, w_bn, X, I["val"], tb)
            v_ens = cham(y[I["val"]], ens)
            v = cham(y[I["val"]], logit_theo_nguon(rieng, ens, may_cua[I["val"]])) if (may_cua[I["val"]] >= 0).any() else v_ens
            them = dict(val_ens_bacc=v_ens["bacc"], **khoang_cach_bn(w_bn))
        else:
            mh.load_state_dict(w_tc)
            v = cham(y[I["val"]], suy_luan(mh, X, I["val"], tb))
        dau = ""
        if v["bacc"] > tot:
            tot, w_tot, dau = v["bacc"], copy.deepcopy(w_tc), " *"
            w_bn_tot = copy.deepcopy(w_bn)
        lich_su.append(dict(vong=vong + 1, lr=lr, buoc=buoc, **huong, **them, **{f"val_{k}": x for k, x in v.items()}))
        nk.ghi(lich_su[-1], vong + 1)
        print(f"vòng {vong+1:>3}/{arg.vong}  lr {lr:.2e}  val bacc {v['bacc']:.4f}  f1 {v['f1_macro']:.4f}  "
              f"acc {v['acc']:.4f}  ({time.time()-t0:.0f}s){dau}", flush=True)

        luu_an_toan(dict(cau_hinh={k: v for k, v in vars(arg).items() if k != "dung_sau_vong"},
                         w=w_tc, vong=vong + 1, tot=tot, w_tot=w_tot, lich_su=lich_su, w_bn=w_bn, w_bn_tot=w_bn_tot,
                         dong_luong=dong_luong, giay=time.time() - t0,
                         rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), f_ckpt)
        if arg.dung_sau_vong and vong + 1 - vong0 >= arg.dung_sau_vong:
            print("(dừng thử theo --dung-sau-vong)")
            return

    # ---- chấm cuối bằng mô hình toàn cục tốt nhất trên val
    tat_ca = np.arange(len(ds))
    if arg.thuat_toan == "fedbn":
        torch.save(dict(w=w_tot, w_bn=w_bn_tot), thu_muc / f"{ten}.pt")
        ens, rieng = du_doan_fedbn(mh, w_tot, w_bn_tot, X, tat_ca, tb)
        logit = logit_theo_nguon(rieng, ens, may_cua)      # train/val theo máy trạm chủ, test theo ensemble
        ket = {p: cham(y[i], logit[i]) for p, i in I.items() if len(i)}
        ket["val_ens"] = cham(y[I["val"]], ens[I["val"]])
        mh.load_state_dict(w_tot)                            # BN trung bình (kiểu FedAvg) — để so sánh
        l_tb = suy_luan(mh, X, I["test"], tb)
        ket["test_bn_tb"] = cham(y[I["test"]], l_tb)
    else:
        mh.load_state_dict(w_tot)
        torch.save(w_tot, thu_muc / f"{ten}.pt")
        logit = suy_luan(mh, X, tat_ca, tb)
        ket = {p: cham(y[i], logit[i]) for p, i in I.items() if len(i)}
    np.savez_compressed(thu_muc / f"{ten}_logit.npz", image=np.asarray(ds.image.tolist(), dtype=np.str_),
                        phan=np.asarray(ds.phan.tolist(), dtype=np.str_), nhan=y, logit=logit.astype(np.float32))
    json.dump(dict(ten=ten, ma_script=ma_file(__file__), ma_du_lieu=ma_file(Path(__file__).parent / "du_lieu.py"),
                   cau_hinh=vars(arg), so_tham_so_mo=so_mo, may_tram=n_may, giay=time.time() - t0, ket_qua=ket, lich_su=lich_su),
              open(f_json, "w"), indent=1, ensure_ascii=False)
    f_ckpt.unlink(missing_ok=True)
    nk.ket_thuc(ket, f_json, dict(giay=time.time() - t0))
    print("\nKẾT QUẢ (mô hình toàn cục tốt nhất trên val):")
    for p, d in ket.items():
        print(f"  {p:<15} bacc {d['bacc']:.4f}  f1 {d['f1_macro']:.4f}  acc {d['acc']:.4f}  auc {d['auc_macro']:.4f}")
    print(f"Đã ghi {f_json}")


if __name__ == "__main__":
    main()
