"""
phan_tich_dem12.py — đọc đêm 12. Không cần GPU.

    python phan_tich_dem12.py

(1) FedBN có lặp lại được trên mô hình BatchNorm thứ hai (ResNet-18) không? So với EfficientNet: +0,080 nguồn thật, +0,013 trộn máy.
(2) Với mô hình LayerNorm (ConvNeXt-atto), cân màu (P2) còn giúp FL không? Dự đoán: KHÔNG, vì đường gây hại qua BatchNorm không tồn tại.
"""
import json

import numpy as np

from du_lieu import RA


def v(thu, ten, k="val"):
    f = RA / thu / f"{ten}.json"
    if not f.exists():
        return np.nan
    kq = json.load(open(f))["ket_qua"]
    return kq[k]["bacc"] if k in kq else np.nan


def tb(xs):
    x = np.array(xs, float); n = int(np.sum(~np.isnan(x)))
    return (f"{np.nanmean(x):.3f}" if n else "  —  "), n


def main():
    print("1. FedBN trên hai mô hình BatchNorm (val; FedBN nguồn = chấm theo nguồn)\n")
    print(f"{'mô hình':<18}{'FedAvg nguồn':>14}{'FedBN nguồn':>13}{'lợi':>8}{'FedAvg trộn':>13}{'FedBN trộn':>12}{'lợi':>8}")
    for bb in ("efficientnet_b0", "resnet18"):
        a = [v("lien_ket", f"fedavg_nguon_{bb}_txlp1_kt224_s{s}") for s in range(3)]
        hau = "_gh" if bb == "efficientnet_b0" else ""
        b = [v("lien_ket", f"fedbn_nguon_{bb}_txlp1{hau}_kt224_s{s}") for s in range(3)]
        c = [v("lien_ket", f"fedavg_tron_nguon_{bb}_txlp1_kt224_s{s}") for s in range(3)]
        d = [v("lien_ket", f"fedbn_tron_nguon_{bb}_txlp1{hau}_kt224_s{s}") for s in range(3)]
        A, B, C, D = (np.nanmean(x) if np.any(~np.isnan(x)) else np.nan for x in (a, b, c, d))
        print(f"{bb:<18}{A:>14.3f}{B:>13.3f}{B - A:>+8.3f}{C:>13.3f}{D:>12.3f}{D - C:>+8.3f}")
    print("\n2. Cân màu (P2) với mô hình LayerNorm (val)\n")
    print(f"{'mô hình':<18}{'tiền xử lý':<11}{'tập trung':>11}{'FedAvg nguồn':>14}{'FedAvg trộn':>13}{'khoảng':>8}")
    for bb in ("efficientnet_b0", "convnext_atto"):
        for p in ("p1", "p2"):
            t = [v("tap_trung", f"{bb}_kt224_nghich_txl{p}_s{s}") for s in range(3)]
            n_ = [v("lien_ket", f"fedavg_nguon_{bb}_txl{p}_kt224_s{s}") for s in range(3)]
            m = [v("lien_ket", f"fedavg_tron_nguon_{bb}_txl{p}_kt224_s{s}") for s in range(3)]
            T, N, M = (np.nanmean(x) if np.any(~np.isnan(x)) else np.nan for x in (t, n_, m))
            print(f"{bb:<18}{p.upper():<11}{T:>11.3f}{N:>14.3f}{M:>13.3f}{T - N:>8.3f}")
    print("\nĐọc kèm: 3 seed, chênh dưới ~0,02 là trong nhiễu. EfficientNet P2 FedAvg nguồn đã có từ đêm 1 (0,683).")


if __name__ == "__main__":
    main()
