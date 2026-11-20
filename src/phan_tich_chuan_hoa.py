"""
phan_tich_chuan_hoa.py — G8b qua kiến trúc: mô hình dùng LayerNorm có ít thiệt hại do máy chụp hơn mô hình BatchNorm không?

    python phan_tich_chuan_hoa.py

Với mỗi backbone (P1, val bAcc, 3 seed): tập trung, FedAvg 4 nguồn thật, FedAvg trộn máy.
  khoảng = tập trung − FedAvg nguồn thật;  phần do máy = (trộn − nguồn) / khoảng  (cận dưới, như với EfficientNet-B0: ~45%).
Dự đoán G8b: phần do máy NHỎ HƠN ở mô hình LayerNorm (ConvNeXt, DeiT), vì chúng không có thống kê BN để bị lấy trung bình.
"""
import json

import numpy as np

from du_lieu import RA

BB = {"efficientnet_b0": "BatchNorm", "resnet18": "BatchNorm", "convnext_atto": "LayerNorm", "deit_tiny_patch16_224": "LayerNorm"}


def doc(f):
    return json.load(open(f))["ket_qua"]["val"]["bacc"] if f.exists() else np.nan


def main():
    print(f"{'backbone':<24}{'chuẩn hoá':<11}{'n':>2}{'tập trung':>11}{'FedAvg nguồn':>14}{'FedAvg trộn':>13}{'khoảng':>8}{'phần do máy':>13}")
    for bb, ch in BB.items():
        t = [doc(RA / "tap_trung" / f"{bb}_kt224_nghich_txlp1_s{s}.json") for s in range(3)]
        n = [doc(RA / "lien_ket" / f"fedavg_nguon_{bb}_txlp1_kt224_s{s}.json") for s in range(3)]
        m = [doc(RA / "lien_ket" / f"fedavg_tron_nguon_{bb}_txlp1_kt224_s{s}.json") for s in range(3)]
        k = int(np.sum(~np.isnan(t)))
        if not k:
            continue
        T, N, M = np.nanmean(t), np.nanmean(n), np.nanmean(m)
        print(f"{bb:<24}{ch:<11}{k:>2}{T:>11.3f}{N:>14.3f}{M:>13.3f}{T - N:>8.3f}{(M - N) / (T - N):>12.0%}")
    print("\nĐọc kèm: với 3 seed, chênh dưới ~0,02 ở từng cột là trong nhiễu.")
    print("\nKIỂM DeiT: trộn máy làm TỆ hơn (0,698 < 0,742) là thật hay do một lần trộn? (3 lần trộn khác nhau)")
    for bb in ("deit_tiny_patch16_224", "convnext_atto"):
        for cot in ("tron_nguon", "tron_nguon1", "tron_nguon2"):
            v = [doc(RA / "lien_ket" / f"fedavg_{cot}_{bb}_txlp1_kt224_s{s}.json") for s in range(3)]
            if int(np.sum(~np.isnan(v))):
                print(f"   {bb:<24}{cot:<13} n={int(np.sum(~np.isnan(v)))}  val {np.nanmean(v):.3f}  (từng seed: {', '.join(f'{x:.3f}' for x in v if not np.isnan(x))})")


if __name__ == "__main__":
    main()
