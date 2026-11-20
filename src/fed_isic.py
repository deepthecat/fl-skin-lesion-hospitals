"""
fed_isic.py — GĐ1: benchmark Fed-ISIC2019 theo ĐÚNG giao thức FLamby (Ogier du Terrail và c.s., NeurIPS 2022),
để so trực tiếp với số đã công bố, rồi chồng phân tích riêng của mình (theo trung tâm, FedBN) lên cùng bộ.

    python fed_isic.py --che-do tap_trung --seed 0
    python fed_isic.py --che-do fedavg   --seed 0
    python fed_isic.py --che-do fedprox  --seed 0          # μ = 0,001 như config của FLamby
    python fed_isic.py --che-do fedbn    --seed 0
    python fed_isic.py --che-do cuc_bo --tram 5 --seed 0    # chỉ huấn luyện trên một trung tâm (đường "Local" của FLamby)

Giao thức chép từ mã gốc FLamby (đã đọc trực tiếp 16.10):
  - 6 trung tâm (0 BCN, 1 molemax, 2 modern, 3 rosendahl, 4 MSK, 5 dias), phép chia train/test của FLamby; KHÔNG có val:
    huấn luyện số bước cố định rồi chấm mô hình cuối trên test (nên không có thiên lệch do chọn mô hình theo val).
  - Ảnh: cạnh ngắn 224 giữ tỉ lệ, colour constancy (bản "cc", mặc định); train cắt ngẫu nhiên 200×200 kèm tăng cường
    albumentations y hệt FLamby; test cắt giữa 200×200. Chuẩn hoá ImageNet.
  - EfficientNet-B0 học trước (gói efficientnet_pytorch như FLamby), đầu ra 8 lớp.
  - Focal loss có trọng số lớp (γ = 2, α như FLamby).
  - Tập trung: Adam lr 5e-4, lô 64, 20 epoch.
  - Liên kết: SGD lr 0,01 tại chỗ, lô 64, mỗi vòng mỗi trung tâm 100 bước; số vòng = get_nb_max_rounds(100) của FLamby
    = (18413 // 6 // 64) * 20 // 100 = 9 vòng. FedAvg gộp theo số ảnh. FedProx μ = 0,001.
  - Độ đo: balanced accuracy. Ghi cả (a) trên toàn bộ test gộp và (b) trung bình bAcc test từng trung tâm,
    vì phải đối chiếu đúng định nghĩa của bảng FLamby trước khi so số.
Khác FLamby (ghi rõ trong bài): dùng mixed precision (AMP) cho vừa GPU 4 GB và nhanh hơn.
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from du_lieu import RA, ma_file
from nhat_ky import NhatKy

ANH = {"cc": Path.home() / "du_lieu_ngoai" / "isic2019_raw" / "isic2019_224_cc",
       "goc": Path.home() / "du_lieu_ngoai" / "isic2019_raw" / "isic2019_224_goc"}
ALPHA = [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
TEN_TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
SZ = 200


# ------------------------------------------------------------------ dữ liệu
def bien_doi(train):
    import albumentations as A
    if train:
        return A.Compose([A.RandomScale(0.07), A.Rotate(50), A.RandomBrightnessContrast(0.15, 0.1), A.Flip(p=0.5),
                          A.Affine(shear=0.1), A.RandomCrop(SZ, SZ), A.CoarseDropout(random.randint(1, 8), 16, 16),
                          A.Normalize(always_apply=True)])
    return A.Compose([A.CenterCrop(SZ, SZ), A.Normalize(always_apply=True)])


class TapAnh(torch.utils.data.Dataset):
    def __init__(self, thu_muc, ten, nhan, train):
        self.thu_muc, self.ten, self.nhan, self.bd = thu_muc, list(ten), np.asarray(nhan), bien_doi(train)

    def __len__(self):
        return len(self.ten)

    def __getitem__(self, i):
        x = np.array(Image.open(self.thu_muc / f"{self.ten[i]}.jpg").convert("RGB"))
        x = self.bd(image=x)["image"]
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), int(self.nhan[i])


def lo(tap, bs, xao, seed, so_luong=4):
    g = torch.Generator(); g.manual_seed(int(seed))
    return torch.utils.data.DataLoader(tap, batch_size=bs, shuffle=xao, num_workers=so_luong, drop_last=False,
                                       persistent_workers=so_luong > 0, generator=g, pin_memory=True)


def vo_han(dl):
    while True:
        for b in dl:
            yield b


# ------------------------------------------------------------------ mô hình, loss
def tao_mo_hinh(pretrained=True, backbone="effnet_flamby"):
    if backbone != "effnet_flamby":                 # xương sống timm (cùng thư viện cho mọi mô hình khi so với nhau)
        import timm
        return timm.create_model(backbone, pretrained=pretrained, num_classes=8)
    from efficientnet_pytorch import EfficientNet
    m = EfficientNet.from_pretrained("efficientnet-b0") if pretrained else EfficientNet.from_name("efficientnet-b0")
    m._fc = nn.Linear(m._fc.in_features, 8)
    m.set_swish(memory_efficient=False)          # an toàn với AMP; cùng giá trị tính toán
    return m


class FocalLoss(nn.Module):
    """BaselineLoss của FLamby: focal loss có trọng số, γ = 2."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        self.gamma = gamma

    def forward(self, z, y):
        logpt = F.log_softmax(z.float(), 1).gather(1, y.view(-1, 1)).view(-1)
        pt = logpt.exp()
        return (-(1 - pt) ** self.gamma * logpt * self.alpha[y]).mean()


def khoa_bn(mh):
    ten = [n for n, m in mh.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    return {k for k in mh.state_dict() if any(k.startswith(n + ".") for n in ten)}


@torch.no_grad()
def suy_luan(mh, dl, tb):
    mh.eval(); ra = []
    for x, _ in dl:
        with torch.autocast(tb.type, enabled=tb.type == "cuda"):
            ra.append(mh(x.to(tb, non_blocking=True)).float().cpu())
    return torch.cat(ra).numpy()


def bacc(y, logit):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, logit.argmax(1)))


def buoc(mh, opt, scaler, crit, x, y, tb, goc=None, mu=0.0):
    x, y = x.to(tb, non_blocking=True), y.to(tb, non_blocking=True)
    with torch.autocast(tb.type, enabled=tb.type == "cuda"):
        loss = crit(mh(x), y)
    if goc is not None:
        loss = loss + (mu / 2) * sum(((p - g) ** 2).sum() for p, g in zip(mh.parameters(), goc))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return float(loss.detach())


# ------------------------------------------------------------------ chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--che-do", required=True, choices=("tap_trung", "fedavg", "fedprox", "fedbn", "cuc_bo"))
    ap.add_argument("--tram", type=int, default=-1, help="trung tâm cho --che-do cuc_bo")
    ap.add_argument("--anh", default="cc", choices=tuple(ANH))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loai-tru", default="", help="CSV cột image: bỏ khỏi train (vd. ảnh nghi trùng HAM)")
    ap.add_argument("--luu-mo-hinh", action="store_true", help="lưu trọng số cuối vào ketqua/mo_hinh/<tên>.pt")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--so-buoc", type=int, default=100, help="số bước tại chỗ mỗi vòng (FLamby: 100)")
    ap.add_argument("--vong", type=int, default=0, help="0 = theo công thức FLamby")
    ap.add_argument("--mu", type=float, default=0.001)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--khong-pretrained", action="store_true")
    ap.add_argument("--gioi-han-anh", type=int, default=0, help="chỉ để thử code: lấy N ảnh mỗi trung tâm")
    ap.add_argument("--ra", default=str(RA / "fed_isic"))
    ap.add_argument("--backbone", default="effnet_flamby", help="effnet_flamby (gói efficientnet_pytorch như FLamby) hoặc tên timm")
    ap.add_argument("--cong-thuc", default="flamby", choices=("flamby", "ham"),
                    help="flamby: Adam 5e-4 (tập trung), SGD/Adam theo --toi-uu-fl (FL) | ham: AdamW 3e-4 + lịch cos như HAM10000")
    ap.add_argument("--kich-ban", default="nguon", choices=("nguon", "tron"), help="client = trung tâm thật | trộn máy (cột may_tron)")
    ap.add_argument("--toi-uu-fl", default="sgd", choices=("sgd", "adam"), help="FL tại chỗ: sgd 0,01 (FLamby) hoặc adam 5e-4")
    arg = ap.parse_args()

    random.seed(arg.seed); np.random.seed(arg.seed); torch.manual_seed(arg.seed)
    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import pandas as pd
    cs = pd.read_csv(RA / "fed_isic_chia.csv")
    if arg.gioi_han_anh:
        cs = cs.groupby(["center", "phan"], group_keys=False).head(arg.gioi_han_anh).reset_index(drop=True)
    tr, te = cs[cs.phan == "train"].reset_index(drop=True), cs[cs.phan == "test"].reset_index(drop=True)
    if arg.loai_tru:
        bo = set(pd.read_csv(arg.loai_tru).image.astype(str)); n0 = len(tr)
        tr = tr[~tr.image.astype(str).isin(bo)].reset_index(drop=True); print(f"Loại trừ {n0 - len(tr)} ảnh train theo {arg.loai_tru}")
    thu_muc = ANH[arg.anh]

    hau = f"_tram{arg.tram}" if arg.che_do == "cuc_bo" else ""
    bbt = "" if arg.backbone == "effnet_flamby" else f"_{arg.backbone}"
    opt_t = "_adam" if (arg.toi_uu_fl == "adam" and arg.che_do not in ("tap_trung", "cuc_bo")) else ""
    kbt = "" if arg.kich_ban == "nguon" else f"_{arg.kich_ban}"
    if arg.kich_ban == "tron" and arg.che_do not in ("fedavg", "fedprox"):
        raise SystemExit("--kich-ban tron chỉ dùng cho fedavg/fedprox")
    ctt = "_ham" if arg.cong_thuc == "ham" else ""
    vbt = f"_v{arg.vong}b{arg.so_buoc}" if (arg.vong or arg.so_buoc != 100) else ""
    vbt += f"_e{arg.epoch}" if (arg.che_do in ("tap_trung", "cuc_bo") and arg.epoch != 20) else ""
    vbt += "_lt" if arg.loai_tru else ""
    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{ctt}{kbt}{vbt}_s{arg.seed}"
    out = Path(arg.ra); out.mkdir(parents=True, exist_ok=True)
    f_json, f_ck = out / f"{ten}.json", out / f"{ten}_ckpt.pt"
    if f_json.exists():
        print(f"Đã có {f_json.name} — bỏ qua."); return

    mh = tao_mo_hinh(not arg.khong_pretrained, arg.backbone).to(tb)
    crit = FocalLoss(ALPHA).to(tb)
    scaler = torch.amp.GradScaler(enabled=tb.type == "cuda")
    dl_test = lo(TapAnh(thu_muc, te.image, te.target, False), 128, False, arg.seed, arg.workers)
    lich_su, t0 = [], time.time()
    nk = NhatKy(ten, vars(arg), "fed_isic", [__file__], them=dict(ma_script=ma_file(__file__), n_train=len(tr), n_test=len(te)))

    def cham_test(logit):
        y = te.target.values
        r = dict(test_gop=bacc(y, logit))
        for c in range(6):
            m = te.center.values == c
            r[f"test_{TEN_TT[c]}"] = bacc(y[m], logit[m]) if m.any() else float("nan")
        r["test_tb_tram"] = float(np.nanmean([r[f"test_{t}"] for t in TEN_TT]))
        return r

    # ---------------- tập trung / cục bộ: Adam lr 5e-4, 20 epoch
    if arg.che_do in ("tap_trung", "cuc_bo"):
        d = tr if arg.che_do == "tap_trung" else tr[tr.center == arg.tram]
        dl = lo(TapAnh(thu_muc, d.image, d.target, True), arg.bs, True, arg.seed, arg.workers)
        opt = (torch.optim.Adam(mh.parameters(), lr=5e-4) if arg.cong_thuc == "flamby"
               else torch.optim.AdamW(mh.parameters(), lr=3e-4, weight_decay=1e-4))
        ep0 = 0
        if f_ck.exists():
            ck = torch.load(f_ck, map_location="cpu", weights_only=False)
            mh.load_state_dict(ck["w"]); opt.load_state_dict(ck["opt"]); ep0, lich_su = ck["ep"], ck["lich_su"]
            print(f"Chạy tiếp từ epoch {ep0 + 1}")
        lich = None
        if arg.cong_thuc == "ham":     # khởi động 10% rồi giảm theo cos (giống OneCycle của HAM10000); không cần lưu trạng thái
            nb, tong = len(dl), arg.epoch * len(dl)
            def he_so(t, nb0=ep0 * nb):
                t += nb0; k = int(0.1 * tong)
                return max(0.04, t / max(k, 1)) if t < k else 0.5 * (1 + math.cos(math.pi * (t - k) / max(tong - k, 1)))
            lich = torch.optim.lr_scheduler.LambdaLR(opt, he_so)
        def mot(x, y):
            r = buoc(mh, opt, scaler, crit, x, y, tb)
            if lich is not None:
                lich.step()
            return r
        for ep in range(ep0, arg.epoch):
            mh.train()
            l = [mot(x, y) for x, y in dl]
            lich_su.append(dict(epoch=ep + 1, loss=float(np.mean(l)), giay=time.time() - t0))
            nk.ghi(lich_su[-1], ep + 1)
            print(f"epoch {ep + 1:>2}/{arg.epoch}  loss {np.mean(l):.4f}  ({time.time() - t0:.0f}s)", flush=True)
            torch.save(dict(w=mh.state_dict(), opt=opt.state_dict(), ep=ep + 1, lich_su=lich_su), f_ck)
        logit = suy_luan(mh, dl_test, tb)
        w_bn = None

    # ---------------- liên kết: SGD lr 0,01, 100 bước/vòng/trung tâm
    else:
        vong = arg.vong or (18413 // 6 // arg.bs) * 20 // arg.so_buoc
        kh = tr.center.values if arg.kich_ban == "nguon" else tr.may_tron.values.astype(int)   # client của từng ảnh train
        tram = sorted(int(c) for c in np.unique(kh))
        dls = {c: vo_han(lo(TapAnh(thu_muc, tr.image[kh == c], tr.target[kh == c], True),
                            arg.bs, True, arg.seed + c, max(0, arg.workers // 2))) for c in tram}
        n_k = {c: int((kh == c).sum()) for c in tram}
        w_tc = copy.deepcopy(mh.state_dict())
        KBN = khoa_bn(mh)
        w_bn = {c: {k: w_tc[k].clone() for k in KBN} for c in tram} if arg.che_do == "fedbn" else None
        v0 = 0
        if f_ck.exists():
            ck = torch.load(f_ck, map_location="cpu", weights_only=False)
            w_tc, w_bn, v0, lich_su = ck["w"], ck["w_bn"], ck["vong"], ck["lich_su"]
            print(f"Chạy tiếp từ vòng {v0 + 1}")
        print(f"{arg.che_do}: {len(tram)} trung tâm {n_k}, {vong} vòng × {arg.so_buoc} bước")
        for v in range(v0, vong):
            cac, ls = {}, []
            for c in tram:
                mh.load_state_dict({**w_tc, **(w_bn[c] if w_bn else {})})
                mh.train()
                if arg.cong_thuc == "ham":
                    opt = torch.optim.AdamW(mh.parameters(), lr=3e-4 * 0.5 * (1 + math.cos(math.pi * v / vong)), weight_decay=1e-4)
                else:
                    opt = (torch.optim.SGD(mh.parameters(), lr=0.01) if arg.toi_uu_fl == "sgd"
                           else torch.optim.Adam(mh.parameters(), lr=5e-4))
                goc = [p.detach().clone() for p in mh.parameters()] if arg.che_do == "fedprox" else None
                for _ in range(arg.so_buoc):
                    x, y = next(dls[c])
                    ls.append(buoc(mh, opt, scaler, crit, x, y, tb, goc, arg.mu))
                cac[c] = {k: t.detach().cpu().clone() for k, t in mh.state_dict().items()}
                if w_bn is not None:
                    w_bn[c] = {k: cac[c][k] for k in KBN}
            tong = sum(n_k.values())
            w_tc = {k: (sum(cac[c][k].float() * n_k[c] / tong for c in tram).to(w_tc[k].dtype)
                        if torch.is_floating_point(w_tc[k]) else cac[tram[0]][k]) for k in w_tc}
            lich_su.append(dict(vong=v + 1, loss=float(np.mean(ls)), giay=time.time() - t0))
            nk.ghi(lich_su[-1], v + 1)
            print(f"vòng {v + 1:>2}/{vong}  loss {np.mean(ls):.4f}  ({time.time() - t0:.0f}s)", flush=True)
            torch.save(dict(w=w_tc, w_bn=w_bn, vong=v + 1, lich_su=lich_su), f_ck)
        if w_bn is None:
            mh.load_state_dict(w_tc); logit = suy_luan(mh, dl_test, tb)
        else:
            # FedBN: ảnh test của trung tâm c chấm bằng mô hình của c (test Fed-ISIC2019 có ghi trung tâm)
            logit = np.zeros((len(te), 8), np.float32)
            for c in tram:
                m = np.flatnonzero(te.center.values == c)
                mh.load_state_dict({**w_tc, **w_bn[c]})
                dl_c = lo(TapAnh(thu_muc, te.image.values[m], te.target.values[m], False), 128, False, arg.seed, min(2, arg.workers))
                logit[m] = suy_luan(mh, dl_c, tb)
            mh.load_state_dict(w_tc)
            ket_bn_tb = cham_test(suy_luan(mh, dl_test, tb))   # BN trung bình, để so

    ket = cham_test(logit)
    if arg.che_do == "fedbn":
        ket.update({f"bn_tb_{k}": v for k, v in ket_bn_tb.items()})
    np.savez_compressed(out / f"{ten}_logit.npz", image=te.image.values.astype(str), center=te.center.values,
                        nhan=te.target.values, logit=logit.astype(np.float32))
    json.dump(dict(ten=ten, ma_script=ma_file(__file__), cau_hinh=vars(arg), giay=time.time() - t0, ket_qua=ket,
                   lich_su=lich_su), open(f_json, "w"), indent=1, ensure_ascii=False)
    if arg.luu_mo_hinh:
        (RA / "mo_hinh").mkdir(exist_ok=True); torch.save(mh.state_dict(), RA / "mo_hinh" / f"{ten}.pt")
        print(f"Đã lưu trọng số: {RA / 'mo_hinh' / (ten + '.pt')}")
    f_ck.unlink(missing_ok=True)
    nk.ket_thuc(ket, f_json, dict(giay=time.time() - t0))
    print("\nKẾT QUẢ (balanced accuracy trên test):")
    print("  gộp toàn bộ test:", f"{ket['test_gop']:.4f}", "| trung bình theo trung tâm:", f"{ket['test_tb_tram']:.4f}")
    print("  " + "  ".join(f"{t} {ket[f'test_{t}']:.3f}" for t in TEN_TT))
    print(f"Đã ghi {f_json}")


if __name__ == "__main__":
    main()
