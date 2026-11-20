"""
phan_tich_5seed.py — các con số CHÍNH của bài với 5 seed thay vì 3, kèm khoảng tin cậy 95% (phân phối t). Không cần GPU.

    python phan_tich_5seed.py
"""
import json

import numpy as np
from scipy import stats

from du_lieu import RA

DONG = [("ConvNeXt-atto tập trung", "tap_trung", "convnext_atto_kt224_nghich_txlp1_s{s}"),
        ("ConvNeXt-atto FedAvg nguồn", "lien_ket", "fedavg_nguon_convnext_atto_txlp1_kt224_s{s}"),
        ("EfficientNet-B0 tập trung", "tap_trung", "efficientnet_b0_kt224_nghich_txlp1_s{s}"),
        ("EfficientNet-B0 FedAvg nguồn", "lien_ket", "fedavg_nguon_efficientnet_b0_txlp1_kt224_s{s}")]
DONG += [(f"EffNet-B0 L0 + DP ε={m} (lr 4)", "dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_efficientnet_b0_txlp1_kt224_s{{s}}") for m in ("8", "3", "1")]


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h, len(x)


def main():
    print(f"{'cấu hình':<34}{'val (3 seed)':>16}{'val (5 seed)':>16}{'test (5 seed)':>16}")
    for nhan, thu, mau in DONG:
        val, test = [], []
        for s in range(5):
            f = RA / thu / (mau.format(s=s) + ".json")
            k = json.load(open(f))["ket_qua"] if f.exists() else None
            val.append(k["val"]["bacc"] if k else np.nan); test.append(k["test"]["bacc"] if k else np.nan)
        val, test = np.array(val), np.array(test)
        m3, h3, _ = ci(val[:3]); m5, h5, n5 = ci(val); t5, ht, _ = ci(test)
        print(f"{nhan:<34}{m3:>9.3f}±{h3:.3f}{m5:>9.3f}±{h5:.3f}{t5:>9.3f}±{ht:.3f}   (n={n5})")
    print("\n±: nửa độ rộng khoảng tin cậy 95% theo seed (phân phối t). Đây là dao động do huấn luyện, chưa gồm dao động của tập chấm.")


if __name__ == "__main__":
    main()
