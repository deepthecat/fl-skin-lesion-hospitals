"""
phan_tich_dem13.py — đọc đêm 13. Không cần GPU.

    python phan_tich_dem13.py

(1) Phép thử sạch cho G8b: CÙNG kiến trúc ResNet-50, CÙNG công thức học trước a1h (ResNet Strikes Back),
    chỉ khác lớp chuẩn hoá: resnet50 (BatchNorm) vs resnet50_gn (GroupNorm — không có "sổ tay chung").
    Dự đoán: BN được lợi khi trộn máy (~+0,05), GN thì ~0; GN ở nguồn thật cao hơn BN.
(2) Kiểm nhiễu đêm 12: FedBN ResNet-18 lợi +0,040 — có lặp lại ở cả 3 seed không?
"""
import json

import numpy as np

from du_lieu import RA

KT = 160   # đêm 13 chạy ảnh 160, lô 32 cho vừa 4 GB (cả hai mô hình như nhau)


def v(thu, ten, k="val"):
    f = RA / thu / f"{ten}.json"
    if not f.exists():
        return np.nan
    kq = json.load(open(f))["ket_qua"]
    return kq[k]["bacc"] if k in kq else np.nan


def m(xs):
    x = np.array(xs, float)
    return np.nanmean(x) if np.any(~np.isnan(x)) else np.nan


def f3(x):
    return "  —  " if np.isnan(x) else f"{x:.3f}"


def main():
    print(f"1. ResNet-50: chỉ đổi lớp chuẩn hoá (val, P1, ảnh {KT}, lô 32)\n")
    print(f"{'mô hình':<14}{'chuẩn hoá':<11}{'tập trung':>10}{'FL nguồn':>10}{'FL trộn':>9}{'khoảng':>8}{'trộn−nguồn':>12}{'phần do máy':>13}")
    for bb, cn in (("resnet50.a1h_in1k", "BatchNorm"), ("resnet50_gn.a1h_in1k", "GroupNorm")):
        t = [v("tap_trung", f"{bb}_kt{KT}_nghich_txlp1_s{s}") for s in range(3)]
        n = [v("lien_ket", f"fedavg_nguon_{bb}_txlp1_kt{KT}_s{s}") for s in range(3)]
        r = [v("lien_ket", f"fedavg_tron_nguon_{bb}_txlp1_kt{KT}_s{s}") for s in range(3)]
        T, N, R = m(t), m(n), m(r)
        phan = (R - N) / (T - N) if T - N > 0 else np.nan
        print(f"{bb.split(".")[0]:<14}{cn:<11}{f3(T):>10}{f3(N):>10}{f3(R):>9}{f3(T - N):>8}{(R - N):>+12.3f}"
              f"{('  —' if np.isnan(phan) else f'{phan:>12.0%}')}")
        print(f"{'':<25}theo seed  tập trung {[f3(x) for x in t]}  nguồn {[f3(x) for x in n]}  trộn {[f3(x) for x in r]}")
    print("\n2. Kiểm nhiễu FedBN trên ResNet-18 (nguồn thật, val)\n")
    for s in range(3):
        a = v("lien_ket", f"fedavg_nguon_resnet18_txlp1_kt224_s{s}")
        b = v("lien_ket", f"fedbn_nguon_resnet18_txlp1_kt224_s{s}")
        print(f"  seed {s}: FedAvg {f3(a)}  FedBN {f3(b)}  lợi {b - a:+.3f}")
    print("\nĐọc kèm: 3 seed, chênh dưới ~0,02 là trong nhiễu. Lợi FedBN 'thật' khi cả 3 seed cùng dấu dương.")


if __name__ == "__main__":
    main()
