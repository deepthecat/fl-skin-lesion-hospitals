"""
phan_tich_theo_site.py — ai trả giá? Tách kết quả theo từng bệnh viện (site) và từng lớp bệnh.

    python phan_tich_theo_site.py

Chỉ đọc logit đã lưu (ketqua/tap_trung/*_logit.npz, ketqua/lien_ket/*_logit.npz) và cột `nguon`
trong ketqua/chia_tap.csv. Không cần GPU, chạy vài giây.

In ra:
  1. Balanced accuracy trên val, tách theo 4 site (test không ghi nguồn nên chỉ tách được trên val).
  2. Recall từng lớp trên test chính thức.
  3. Tỉ lệ ảnh test bị đoán là NV (cột NV_doan_test; thật ra NV chiếm khoảng 60% test) — mô hình nghiêng về NV bao nhiêu (molemax có 94% NV).
Ghi ketqua/theo_site.csv và ketqua/theo_lop.csv.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from du_lieu import LOP, RA

warnings.filterwarnings("ignore")
NGUON = ["vidir_molemax", "vidir_modern", "rosendahl", "vienna_dias"]
NGAN = {"vidir_molemax": "molemax", "vidir_modern": "modern", "rosendahl": "rosendahl", "vienna_dias": "dias"}

CAU_HINH = {   # tên hiển thị -> (thư mục, mẫu tên file với {s} là seed)
    "tập trung P1": ("tap_trung", "efficientnet_b0_kt224_nghich_txlp1_s{s}"),
    "tập trung P2": ("tap_trung", "efficientnet_b0_kt224_nghich_txlp2_s{s}"),
}
for tt in ("fedavg", "fedprox"):
    for p in ("p1", "p2"):
        for cb, nhan in (("", "toàn cục"), ("_cb", "cục bộ")):
            CAU_HINH[f"{tt} {p.upper()} w:{nhan}"] = ("lien_ket", f"{tt}_nguon_efficientnet_b0_txl{p}{cb}_kt224_s{{s}}")
for db in ("k2", "k1", "dau"):   # thí nghiệm 14.10: đóng băng thân mô hình
    CAU_HINH[f"fedavg P1 đb:{db}"] = ("lien_ket", f"fedavg_nguon_efficientnet_b0_db{db}_txlp1_kt224_s{{s}}")
TRON = ("tron_nguon", "tron_nguon1", "tron_nguon2")   # đối chứng: giữ kích thước + nhãn, trộn máy chụp; 3 lần trộn khác nhau
for kb in TRON:
    for p in ("p1", "p2"):
        CAU_HINH[f"fedavg {p.upper()} {kb}"] = ("lien_ket", f"fedavg_{kb}_efficientnet_b0_txl{p}_kt224_s{{s}}")


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f}±{x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def main():
    cs = pd.read_csv(RA / "chia_tap.csv")
    if "nguon" not in cs.columns:
        raise SystemExit("chia_tap.csv chưa có cột nguon — chạy notebook 06 trước.")
    nguon_cua = dict(zip(cs.image, cs.nguon))

    hang_site, hang_lop = [], []
    for ten, (thu_muc, mau) in CAU_HINH.items():
        for s in (0, 1, 2):
            f = RA / thu_muc / f"{mau.format(s=s)}_logit.npz"
            if not f.exists():
                continue
            d = np.load(f)
            yh, y, phan = d["logit"].argmax(1), d["nhan"], d["phan"]
            ng = np.array([nguon_cua.get(i, "") for i in d["image"]])
            v = phan == "val"
            r = dict(cau_hinh=ten, seed=s, val_tong=balanced_accuracy_score(y[v], yh[v]))
            for n in NGUON:
                m = v & (ng == n)
                r[NGAN[n]] = balanced_accuracy_score(y[m], yh[m]) if m.any() else np.nan
            t = phan == "test"
            r["NV_doan_test"] = float(np.mean(yh[t] == LOP.index("NV")))
            hang_site.append(r)
            l = dict(cau_hinh=ten, seed=s)
            for c, ten_lop in enumerate(LOP):
                mc = t & (y == c)
                l[ten_lop] = float(np.mean(yh[mc] == c)) if mc.any() else np.nan
            hang_lop.append(l)

    if not hang_site:
        raise SystemExit("Không tìm thấy file *_logit.npz nào khớp tên — kiểm tra ketqua/tap_trung và ketqua/lien_ket.")
    S, L = pd.DataFrame(hang_site), pd.DataFrame(hang_lop)
    S.to_csv(RA / "theo_site.csv", index=False)
    L.to_csv(RA / "theo_lop.csv", index=False)

    cot = ["val_tong"] + list(NGAN.values()) + ["NV_doan_test"]
    print("1. VAL BALANCED ACCURACY THEO SITE (trung bình ± độ lệch chuẩn qua seed)")
    print("   số ảnh val: molemax 559, modern 466, rosendahl 333, dias 74\n")
    print(f"{'cấu hình':<26}{'n':>2}" + "".join(f"{c:>15}" for c in cot))
    for ten, g in S.groupby("cau_hinh", sort=False):
        print(f"{ten:<26}{len(g):>2}" + "".join(f"{ms(g[c].dropna()):>15}" for c in cot))

    print("\n2. RECALL TỪNG LỚP TRÊN TEST CHÍNH THỨC (trung bình qua seed)\n")
    print(f"{'cấu hình':<26}{'n':>2}" + "".join(f"{c:>8}" for c in LOP))
    for ten, g in L.groupby("cau_hinh", sort=False):
        print(f"{ten:<26}{len(g):>2}" + "".join(f"{g[c].mean():>8.3f}" for c in LOP))

    # khoảng cách tập trung - federated theo site, cùng kiểu tiền xử lý, trọng số toàn cục
    print("\n3. KHOẢNG CÁCH TẬP TRUNG − FEDAVG THEO SITE (val, trọng số toàn cục)\n")
    print(f"{'tiền xử lý':<26}" + "".join(f"{c:>15}" for c in ["val_tong"] + list(NGAN.values())))
    for p in ("P1", "P2"):
        a = S[S.cau_hinh == f"tập trung {p}"]
        for kb, ten_b in [("nguon", f"fedavg {p} w:toàn cục")] + [(t, f"fedavg {p} {t}") for t in TRON]:
            b = S[S.cau_hinh == ten_b]
            if len(a) and len(b):
                print(f"{p + ' ' + kb:<26}" + "".join(f"{a[c].mean() - b[c].mean():>15.3f}" for c in ["val_tong"] + list(NGAN.values())))
    print("   (site ở đây là nguồn gốc thật của ảnh val; val không bị trộn nên hai dòng so được trực tiếp)")
    print(f"\nĐã ghi {RA / 'theo_site.csv'} và {RA / 'theo_lop.csv'}")


if __name__ == "__main__":
    main()
