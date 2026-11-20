"""
recall_theo_lop.py — recall từng lớp (đặc biệt MEL) của các mô hình chính, từ logit đã lưu. Chỉ CPU, vài giây.

    cd ~/ham_fl && source .venv/bin/activate && python recall_theo_lop.py

Đọc ketqua/{tap_trung,lien_ket}/<tên>_logit.npz (đã có sẵn từ các lần chạy), không train lại gì.
Ra: bảng trên màn hình + ketqua/recall_theo_lop.csv (từng seed) để đưa vào paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RA = Path.home() / "ham_fl" / "ketqua"
LOP = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MO_HINH = {
    "EfficientNet-B0 tập trung": "tap_trung/efficientnet_b0_kt224_nghich_txlp1_s{s}",
    "EfficientNet-B0 FedAvg thật": "lien_ket/fedavg_nguon_efficientnet_b0_txlp1_kt224_s{s}",
    "EfficientNet-B0 FedAvg trộn": "lien_ket/fedavg_tron_nguon_efficientnet_b0_txlp1_kt224_s{s}",
    "ConvNeXt-atto tập trung": "tap_trung/convnext_atto_kt224_nghich_txlp1_s{s}",
    "ConvNeXt-atto FedAvg thật": "lien_ket/fedavg_nguon_convnext_atto_txlp1_kt224_s{s}",
    "ConvNeXt-atto FedAvg trộn": "lien_ket/fedavg_tron_nguon_convnext_atto_txlp1_kt224_s{s}",
}

hang = []
for nhan, mau in MO_HINH.items():
    for s in range(3):
        f = RA / (mau.format(s=s) + "_logit.npz")
        if not f.exists():
            print(f"THIẾU: {f}")
            continue
        d = np.load(f)
        yh = d["logit"].argmax(1)
        for phan in ("val", "test"):
            m = d["phan"] == phan
            y, p = d["nhan"][m], yh[m]
            r = dict(mo_hinh=nhan, seed=s, phan=phan, bacc=np.mean([np.mean(p[y == k] == k) for k in range(7) if (y == k).any()]))
            for k, ten in enumerate(LOP):
                r[ten] = float(np.mean(p[y == k] == k)) if (y == k).any() else np.nan
            r["ti_le_doan_NV"] = float(np.mean(p == LOP.index("NV")))
            hang.append(r)

b = pd.DataFrame(hang)
b.to_csv(RA / "recall_theo_lop.csv", index=False)
pd.set_option("display.width", 200)
for phan in ("test", "val"):
    print(f"\n=== Recall từng lớp trên {phan}, trung bình 3 seed ===")
    print(b[b.phan == phan].groupby("mo_hinh", sort=False)[["bacc"] + LOP + ["ti_le_doan_NV"]].mean().round(3).to_string())
print(f"\nĐã ghi {RA / 'recall_theo_lop.csv'}")
