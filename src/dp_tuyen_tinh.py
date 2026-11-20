"""
dp_tuyen_tinh.py — RQ2 phần 2, làm lại: DP-SGD chỉ trên TẦNG PHÂN LOẠI CUỐI, đặc trưng đóng băng.

    python dp_tuyen_tinh.py --luoi               # cả lưới: 4 kịch bản × 6 mức riêng tư × 3 seed
    python dp_tuyen_tinh.py --kich-ban iid --epsilon 3 --seed 0

VÌ SAO LÀM LẠI. Lần trước huấn luyện cả ResNet18 bằng DP-SGD: mọi lượt ε=8/3/1 sập về bacc 0.143 (= đoán
một lớp). Đó là cấu hình hỏng, không phải giá của DP: Adam + nhiễu làm 11 triệu tham số trôi ngẫu nhiên;
đổi BatchNorm sang GroupNorm đã làm hỏng phần pretrained ngay cả khi KHÔNG có DP (0.549 so với 0.720);
lô 64 ảnh quá nhỏ.

CÁCH ĐÚNG (Tramèr & Boneh, ICLR 2021, "Differentially Private Learning Needs Better Features"):
  1. Lấy một mô hình pretrained trên ImageNet — dữ liệu CÔNG KHAI — đóng băng, chạy qua mọi ảnh một lần,
     lấy vector đặc trưng. Không dùng mô hình đã fine-tune trên HAM10000: nó đã "thấy" dữ liệu bệnh nhân
     mà không có DP, dùng nó là phá bảo đảm.
  2. Chỉ huấn luyện một tầng tuyến tính (đặc trưng -> 7 lớp) bằng DP-SGD.
  Vài nghìn tham số thay vì 11 triệu, nên nhiễu không nhấn chìm tín hiệu; BatchNorm đóng băng thì không
  trộn thông tin giữa các ảnh nên không cần đổi; và chạy nhanh tới mức cả lưới mất vài phút.

DP-SGD ở đây VIẾT TAY (không qua Opacus) vì với tầng tuyến tính, gradient riêng từng ảnh có công thức
đóng: với ảnh i có đặc trưng x_i (đã chuẩn hoá độ dài 1) và sai số g_i = w_y·(p_i − onehot(y_i)),
  gradient theo W = x_i ⊗ g_i, theo b = g_i, độ dài = |g_i|·sqrt(|x_i|² + 1).
Mỗi bước: lấy mẫu Poisson (mỗi ảnh vào lô với xác suất q), cắt từng gradient về độ dài ≤ C, cộng lại,
thêm nhiễu Gauss N(0, (σC)²) vào từng phần tử, chia cho kích thước lô kỳ vọng. Ngân sách ε tính bằng bộ
kế toán RDP của Opacus — phần đó vẫn dùng thư viện vì đó là chỗ dễ sai nhất.

Sáu mức riêng tư:
  inf  — không cắt, không nhiễu (mốc).
  clip — chỉ cắt, không nhiễu. Tách giá của việc CẮT khỏi giá của NHIỄU.
  8, 3, 1, 0.5 — DP-SGD đầy đủ.

Learning rate: chọn MỘT lần bằng nhánh tập trung không DP trên validation, rồi dùng cho mọi nhánh. Việc
chọn này không tính vào ngân sách ε — ghi rõ trong bài.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nhat_ky import NhatKy
from du_lieu import DL, LOP, RA, TB, cham, doc_chia_tap, ma_file, nap_dem, tao_mo_hinh
from huan_luyen_tap_trung import trong_so_lop

MUC = ("inf", "clip", "8", "3", "1", "0.5")
KICH_BAN = ("mot_may", "iid", "dir0.3", "dir0.1")


# ---------------------------------------------------------------- đặc trưng
@torch.no_grad()
def dac_trung(ten, kt, ds, tb, pretrained=True, kieu="p0", ts="", nhan=""):
    duoi = "" if kieu == "p0" else f"_{kieu}"
    duoi += f"_{nhan}" if nhan else ""
    f = RA / f"dac_trung_{ten}_{kt}{duoi}.npy"
    if f.exists():
        return np.load(f)
    print(f"Trích đặc trưng {ten} {kt}px kiểu {kieu} (ImageNet, đóng băng) cho {len(ds)} ảnh...", flush=True)
    X = nap_dem(ds, kt, kieu)
    mh = tao_mo_hinh(ten, pretrained=pretrained)
    if ts:                                      # xương sống học trước trên dữ liệu công khai (bỏ tầng phân loại cũ)
        sd = {k: v for k, v in torch.load(ts, map_location="cpu").items() if not k.startswith(("classifier.", "head.fc.", "fc."))}
        r = mh.load_state_dict(sd, strict=False)
        thieu = [k for k in r.missing_keys if not k.startswith(("classifier.", "head.fc.", "fc."))]
        print(f"Nạp {ts}: thiếu {len(thieu)} khoá, thừa {len(r.unexpected_keys)} khoá", flush=True)
        if thieu or r.unexpected_keys:
            raise SystemExit(f"LỖI nạp xương sống: thiếu {thieu[:5]} thừa {r.unexpected_keys[:5]}")
    mh.reset_classifier(0)                      # bỏ tầng phân loại -> trả vector đặc trưng
    mh = mh.to(tb).eval()
    tb_, dl_ = torch.tensor(TB, device=tb).view(1, 3, 1, 1), torch.tensor(DL, device=tb).view(1, 3, 1, 1)
    ra = []
    for i in range(0, len(X), 32):
        x = torch.from_numpy(X[i:i + 32]).to(tb).permute(0, 3, 1, 2).float().div_(255)
        with torch.autocast(tb.type, enabled=tb.type == "cuda"):
            ra.append(mh((x - tb_) / dl_).float().cpu())
    F_ = torch.cat(ra).numpy()
    np.save(f, F_)
    return F_


# ---------------------------------------------------------------- DP-SGD tay cho tầng tuyến tính
def gradient_tung_anh(W, b, x, y, w_lop):
    """Trả về g (n×K) sao cho gradient theo W của ảnh i là x_i ⊗ g_i, theo b là g_i."""
    p = torch.softmax(x @ W + b, dim=1)
    g = p.clone()
    g[torch.arange(len(y)), y] -= 1.0
    return g * w_lop[y].unsqueeze(1)


def cuc_bo(W, b, x, y, w_lop, arg, q, buoc, sigma, cat, gen, lr):
    """buoc bước DP-SGD trên dữ liệu (x, y) của một máy trạm. Trả về W, b mới."""
    W, b = W.clone(), b.clone()
    vW, vb = torch.zeros_like(W), torch.zeros_like(b)
    n = len(y)
    B_ky_vong = max(q * n, 1.0)
    for _ in range(buoc):
        m = torch.rand(n, generator=gen, device=x.device) < q
        if m.any():
            xb, yb = x[m], y[m]
            g = gradient_tung_anh(W, b, xb, yb, w_lop)
            if cat:
                do_dai = g.norm(dim=1) * torch.sqrt((xb ** 2).sum(1) + 1.0)
                g = g * torch.clamp(arg.clip / (do_dai + 1e-12), max=1.0).unsqueeze(1)
            GW, Gb = xb.T @ g, g.sum(0)
        else:
            GW, Gb = torch.zeros_like(W), torch.zeros_like(b)
        if sigma > 0:
            GW = GW + torch.randn(W.shape, generator=gen, device=x.device) * sigma * arg.clip
            Gb = Gb + torch.randn(b.shape, generator=gen, device=x.device) * sigma * arg.clip
        GW, Gb = GW / B_ky_vong, Gb / B_ky_vong
        vW = 0.9 * vW + GW
        vb = 0.9 * vb + Gb
        W -= lr * vW
        b -= lr * vb
    return W, b


def sigma_cho(muc, delta, q, buoc):
    if muc in ("inf", "clip"):
        return 0.0, math.inf
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    s = float(get_noise_multiplier(target_epsilon=float(muc), target_delta=delta, sample_rate=q, steps=buoc,
                                   accountant="rdp"))
    a = RDPAccountant(); a.history = [(s, q, buoc)]
    return s, float(a.get_epsilon(delta))


def chay(F_all, ds, kich_ban, muc, seed, arg, lr, tb, ghi=True):
    y_all = torch.from_numpy(ds.nhan.values).long().to(tb)
    x_all = torch.from_numpy(F_all).float().to(tb)
    x_all = x_all / x_all.norm(dim=1, keepdim=True).clamp_min(1e-12)       # chuẩn hoá từng ảnh: không tốn ε
    I = {p: np.flatnonzero(ds.phan.values == p) for p in ("train", "val", "val_chinh_thuc", "test")}
    if "trung_train" in ds.columns:
        I["test_sach"] = np.flatnonzero((ds.phan.values == "test") & (ds.trung_train.values == 0))
    if kich_ban == "mot_may":
        may = {0: I["train"]}
    else:
        m = ds[f"may_{kich_ban}"].values
        may = {int(k): np.flatnonzero((ds.phan.values == "train") & (m == k)) for k in sorted(np.unique(m[I["train"]]))}
    w = trong_so_lop(ds.nhan.values[I["train"]], arg.trong_so)
    w_lop = (w if w is not None else torch.ones(len(LOP))).to(tb)

    tt = {}
    for k, idx in may.items():
        q = min(1.0, arg.bs / len(idx))
        buoc_vong = math.ceil(1 / q) * arg.epoch_cuc_bo
        s, e = sigma_cho(muc, arg.delta, q, buoc_vong * arg.vong)
        tt[k] = dict(n=len(idx), q=q, buoc_moi_vong=buoc_vong, sigma=s, eps_that=e)

    gen = torch.Generator(device=tb).manual_seed(seed)
    torch.manual_seed(seed)
    d = x_all.shape[1]
    W, b = torch.zeros(d, len(LOP), device=tb), torch.zeros(len(LOP), device=tb)
    yv = ds.nhan.values[I["val"]]
    tot, Wt, bt, lich_su = -1.0, W, b, []
    for vong in range(arg.vong):
        cac, n_k = [], []
        for k, idx in may.items():
            t = tt[k]
            Wk, bk = cuc_bo(W, b, x_all[idx], y_all[idx], w_lop, arg, t["q"], t["buoc_moi_vong"], t["sigma"],
                            cat=muc != "inf", gen=gen, lr=lr)
            cac.append((Wk, bk)); n_k.append(len(idx))
        tong = sum(n_k)
        W = sum(Wk * (n / tong) for (Wk, _), n in zip(cac, n_k))
        b = sum(bk * (n / tong) for (_, bk), n in zip(cac, n_k))
        v = cham(yv, (x_all[I["val"]] @ W + b).cpu().numpy())
        lich_su.append(dict(vong=vong + 1, val_bacc=v["bacc"]))
        if v["bacc"] > tot:
            tot, Wt, bt = v["bacc"], W.clone(), b.clone()
    logit = (x_all @ Wt + bt).cpu().numpy()
    ket = {p: cham(ds.nhan.values[i], logit[i]) for p, i in I.items() if len(i)}
    if ghi:
        txl = "" if arg.tien_xu_ly == "p0" else f"_txl{arg.tien_xu_ly}"
        vg = f"_v{arg.vong}" if arg.vong != 30 else ""
        ten = f"dptt_{kich_ban}_eps{muc}_{arg.backbone}{'_' + arg.nhan_dt if arg.nhan_dt else ''}{txl}{vg}_kt{arg.kt}_s{seed}"
        ten_nk = ten + (f"_lr{lr:g}" if arg.lr > 0 else "")   # wandb: lr cố định có id riêng (thư mục khác vẫn cùng tên file)
        thu_muc = Path(arg.ra); thu_muc.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(thu_muc / f"{ten}_logit.npz", image=np.asarray(ds.image.tolist(), dtype=np.str_),
                            phan=np.asarray(ds.phan.tolist(), dtype=np.str_), nhan=ds.nhan.values,
                            logit=logit.astype(np.float32))
        cau_hinh = dict(vars(arg), kich_ban=kich_ban, seed=seed, lr=lr, epsilon=muc)
        json.dump(dict(ten=ten, ma_script=ma_file(__file__), epsilon=muc, cau_hinh=cau_hinh,
                       may_tram={str(k): v for k, v in tt.items()}, ket_qua=ket, lich_su=lich_su),
                  open(thu_muc / f"{ten}.json", "w"), indent=1, ensure_ascii=False, default=str)
        nk = NhatKy(ten_nk, {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in cau_hinh.items()},
                    "dp_tuyen_tinh", [__file__],
                    them=dict(sigma=str([round(t["sigma"], 3) for t in tt.values()]), ma_script=ma_file(__file__)))
        for h in lich_su:
            nk.ghi(h, h["vong"])
        nk.ket_thuc(ket, thu_muc / f"{ten}.json")
    return ket, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luoi", action="store_true", help="chạy cả lưới kịch bản × mức riêng tư × seed")
    ap.add_argument("--kich-ban", default="iid")
    ap.add_argument("--cac-kich-ban", default=",".join(KICH_BAN),
                    help="danh sách kịch bản cho --luoi, cách nhau dấu phẩy (vd mot_may,nguon,tron_nguon)")
    ap.add_argument("--tien-xu-ly", default="p0", choices=("p0", "p1", "p2"))
    ap.add_argument("--epsilon", default="3", help="inf, clip, hoặc một số")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trong-so-backbone", default="", help="file .pt trọng số xương sống (timm) học trước")
    ap.add_argument("--nhan-dt", default="", help="nhãn đặc trưng, vd. bcn")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--kt", type=int, default=224)
    ap.add_argument("--vong", type=int, default=30)
    ap.add_argument("--epoch-cuc-bo", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256, help="lô kỳ vọng; DP cần lô lớn")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = tự chọn trên val bằng nhánh tập trung không DP")
    ap.add_argument("--trong-so", default="nghich", choices=("nghich", "can", "khong"))
    ap.add_argument("--khong-pretrained", action="store_true", help="chỉ để thử code")
    ap.add_argument("--ra", default=str(RA / "dp_tuyen_tinh"))
    arg = ap.parse_args()

    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    F_all = dac_trung(arg.backbone, arg.kt, ds, tb, pretrained=not arg.khong_pretrained, kieu=arg.tien_xu_ly, ts=arg.trong_so_backbone, nhan=arg.nhan_dt)
    print(f"Đặc trưng: {F_all.shape} | {tb}")

    lr = arg.lr
    if lr <= 0:
        print("Chọn learning rate (tập trung, không DP, theo val bacc):")
        tot = -1
        for thu in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            ket, _ = chay(F_all, ds, "mot_may", "inf", 0, arg, thu, tb, ghi=False)
            print(f"   lr {thu:<5} val bacc {ket['val']['bacc']:.4f}")
            if ket["val"]["bacc"] > tot:
                tot, lr = ket["val"]["bacc"], thu
        print(f"-> lr = {lr}")

    viec = ([(kb, m, s) for s in (0, 1, 2) for kb in arg.cac_kich_ban.split(",") for m in MUC] if arg.luoi
            else [(arg.kich_ban, arg.epsilon, arg.seed)])
    t0 = time.time()
    for kb, m, s in viec:
        ket, tt = chay(F_all, ds, kb, m, s, arg, lr, tb)
        sig = " ".join(f"{v['sigma']:.2f}" for v in tt.values())
        print(f"{kb:<8} ε={m:<5} s{s}  test bacc {ket['test']['bacc']:.4f}  val {ket['val']['bacc']:.4f}  "
              f"auc {ket['test']['auc_macro']:.4f}  σ=[{sig}]  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nĐã ghi vào {arg.ra}")


if __name__ == "__main__":
    main()
