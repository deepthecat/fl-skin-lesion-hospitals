"""
tan_cong_theo_site.py — giả thuyết H2: rủi ro bị lộ "có trong dữ liệu huấn luyện" có chia đều cho mọi bệnh viện
và mọi loại bệnh không?

    python tan_cong_theo_site.py

Chỉ đọc logit đã lưu, không cần GPU, chạy vài giây.

Tấn công giống tan_cong_thanh_vien.py (ngưỡng loss, Yeom 2018): loss thấp -> đoán "là thành viên".
Khác ở chỗ ta TÁCH theo nguồn: với mỗi bệnh viện, thành viên = ảnh train CỦA bệnh viện đó,
không thành viên = ảnh val CỦA bệnh viện đó. Hai phía cùng nguồn, nên kẻ tấn công không thể
"ăn gian" nhờ khác biệt máy chụp — con số chỉ phản ánh mô hình nhớ bài bao nhiêu.

In ra AUC tấn công (0.5 = kín) theo site và theo lớp, trung bình qua seed, cho:
tập trung, FedAvg trên 4 nguồn thật, FedAvg trên kịch bản trộn máy.
Lưu ý: dias chỉ có 74 ảnh val -> con số của dias dao động mạnh, đọc kèm độ lệch chuẩn.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from du_lieu import LOP, RA
from phan_tich_theo_site import NGAN, NGUON
from tan_cong_thanh_vien import loss_tung_anh

warnings.filterwarnings("ignore")
CAU_HINH = {}
for p in ("p1", "p2"):
    P = p.upper()
    CAU_HINH[f"tập trung {P}"] = ("tap_trung", f"efficientnet_b0_kt224_nghich_txl{p}_s{{s}}")
    CAU_HINH[f"fedavg {P} nguon"] = ("lien_ket", f"fedavg_nguon_efficientnet_b0_txl{p}_kt224_s{{s}}")
    CAU_HINH[f"fedavg {P} tron_nguon"] = ("lien_ket", f"fedavg_tron_nguon_efficientnet_b0_txl{p}_kt224_s{{s}}")
for ts in ("can", "khong"):   # thí nghiệm 13.10: trọng số lớp có làm lớp hiếm lộ nhiều hơn không?
    CAU_HINH[f"tập trung P1 w:{ts}"] = ("tap_trung", f"efficientnet_b0_kt224_{ts}_txlp1_s{{s}}")
# thí nghiệm 14.10: đóng băng thân mô hình như một biện pháp riêng tư (không DP)
CAU_HINH["1 máy P1 (30 vòng)"] = ("lien_ket", "fedavg_mot_may_efficientnet_b0_txlp1_kt224_s{s}")
for db in ("k2", "k1", "dau"):
    for kb, nhan in (("mot_may", "1 máy"), ("nguon", "fedavg")):
        CAU_HINH[f"{nhan} P1 đb:{db}"] = ("lien_ket", f"fedavg_{kb}_efficientnet_b0_db{db}_txlp1_kt224_s{{s}}")


def auc_an_toan(la_tv, diem):
    return roc_auc_score(la_tv, diem) if 0 < la_tv.sum() < len(la_tv) else np.nan


def ms(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return "—"
    return f"{x.mean():.3f}±{x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def ve_hinh(nguon_cua):
    """Hình H4: phân bố loss của ảnh ĐÃ HỌC (train) và ảnh CHƯA THẤY (val), seed 0.
    Hai đống càng lệch nhau, kẻ tấn công càng dễ đặt một vạch ngưỡng để tách chúng."""
    ve = [("tập trung P1", "tap_trung", "efficientnet_b0_kt224_nghich_txlp1_s0"),
          ("FedAvg P1, 4 nguồn thật", "lien_ket", "fedavg_nguon_efficientnet_b0_txlp1_kt224_s0")]
    ve = [v for v in ve if (RA / v[1] / f"{v[2]}_logit.npz").exists()]
    if not ve:
        return
    fig, ax = plt.subplots(1, len(ve), figsize=(6 * len(ve), 3.8), squeeze=False)
    for a, (ten, tm, f) in zip(ax[0], ve):
        d = np.load(RA / tm / f"{f}_logit.npz")
        phan, y, lg = d["phan"], d["nhan"], d["logit"].astype(np.float64)
        L = np.clip(loss_tung_anh(lg, y), 1e-6, None)   # ảnh đoán gần như chắc chắn đúng có loss ~0
        bins = np.logspace(-6, 1.5, 60)
        a.hist(L[phan == "train"], bins=bins, density=True, alpha=0.55, label="đã học (train)")
        a.hist(L[phan == "val"], bins=bins, density=True, alpha=0.55, label="chưa thấy (val)")
        a.set_xscale("log"); a.set_xlabel("loss của từng ảnh (thang log)"); a.set_title(ten)
        a.legend()
    fig.suptitle("Mô hình 'thuộc bài': ảnh đã học có loss thấp hơn ảnh chưa thấy", y=1.02)
    win = Path("/mnt/c/Users/admin/Downloads/Research_2026_Code")
    for dd in [RA / "hinh_hoc"] + ([win / "hinh_hoc"] if win.exists() else []):
        dd.mkdir(parents=True, exist_ok=True)
        fig.savefig(dd / "H4_loss_thanh_vien.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Đã vẽ hinh_hoc/H4_loss_thanh_vien.png\n")


def main():
    cs = pd.read_csv(RA / "chia_tap.csv")
    nguon_cua = dict(zip(cs.image, cs.nguon))
    ve_hinh(nguon_cua)
    hang = []
    for ten, (thu_muc, mau) in CAU_HINH.items():
        for s in (0, 1, 2):
            f = RA / thu_muc / f"{mau.format(s=s)}_logit.npz"
            if not f.exists():
                continue
            d = np.load(f)
            phan, y, logit = d["phan"], d["nhan"], d["logit"].astype(np.float64)
            ng = np.array([nguon_cua.get(i, "") for i in d["image"]])
            chon = (phan == "train") | (phan == "val")
            diem = -loss_tung_anh(logit[chon], y[chon])
            la_tv = (phan[chon] == "train").astype(int)
            ng, y = ng[chon], y[chon]
            r = dict(cau_hinh=ten, seed=s, tong=auc_an_toan(la_tv, diem))
            for n in NGUON:
                m = ng == n
                r[NGAN[n]] = auc_an_toan(la_tv[m], diem[m])
            for c, l in enumerate(LOP):
                m = y == c
                r[l] = auc_an_toan(la_tv[m], diem[m])
            hang.append(r)
    if not hang:
        raise SystemExit("Không tìm thấy file *_logit.npz nào khớp tên.")
    B = pd.DataFrame(hang)
    B.to_csv(RA / "tan_cong_theo_site.csv", index=False)

    cot = ["tong"] + list(NGAN.values())
    print("1. AUC TẤN CÔNG SUY LUẬN THÀNH VIÊN THEO BỆNH VIỆN (0.5 = kín; càng cao càng lộ)")
    print("   số ảnh train/val: molemax ~2830/559, modern ~2440/466, rosendahl ~1590/333, dias ~290/74\n")
    print(f"{'cấu hình':<24}{'n':>2}" + "".join(f"{c:>14}" for c in cot))
    for ten, g in B.groupby("cau_hinh", sort=False):
        print(f"{ten:<24}{len(g):>2}" + "".join(f"{ms(g[c]):>14}" for c in cot))

    print("\n2. AUC TẤN CÔNG THEO LOẠI BỆNH (trung bình qua seed)\n")
    print(f"{'cấu hình':<24}{'n':>2}" + "".join(f"{l:>8}" for l in LOP))
    for ten, g in B.groupby("cau_hinh", sort=False):
        print(f"{ten:<24}{len(g):>2}" + "".join(f"{g[l].mean():>8.3f}" for l in LOP))
    print(f"\nĐã ghi {RA / 'tan_cong_theo_site.csv'}")


if __name__ == "__main__":
    main()
