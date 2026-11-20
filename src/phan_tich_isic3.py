"""
phan_tich_isic3.py — phép đối chứng TRỘN MÁY trên Fed-ISIC2019: G8b có lặp lại trên bộ dữ liệu thứ hai không? Không cần GPU.

    python phan_tich_isic3.py

Cùng mô hình (EfficientNet-B0 bản FLamby), cùng giao thức FLamby (SGD 0,01, 100 bước/vòng, 9 vòng), chỉ đổi cách chia client:
  nguồn thật (6 trung tâm)  vs  trộn máy (giữ số ảnh từng lớp ở mỗi client, trộn máy chụp).
Phần khoảng cách biến mất khi trộn = phần do máy chụp. Trên HAM10000 phần này là ~40% với EfficientNet.
Dự đoán (viết trước): trộn máy làm FedAvg tăng > 0,03 ở cả 3 seed (MSK và rosendahl tăng nhiều nhất).
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")

TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]


def doc(ten):
    return [json.load(open(f))["ket_qua"] if (f := RA / "fed_isic" / f"{ten}_s{s}.json").exists() else None for s in range(3)]


def v(rs, k):
    return np.array([r[k] if r else np.nan for r in rs], float)


def main():
    tt, ng, tr = doc("isic_tap_trung_cc"), doc("isic_fedavg_cc"), doc("isic_fedavg_cc_tron")
    T, N, M = (np.nanmean(v(x, "test_tb_tram")) for x in (tt, ng, tr))
    print("Fed-ISIC2019, EfficientNet-B0 (FLamby), bAcc trung bình đều 6 trung tâm, 3 seed\n")
    print(f"   tập trung {T:.3f} | FedAvg nguồn thật {N:.3f} | FedAvg trộn máy {M:.3f}")
    print(f"   khoảng {T - N:.3f} | trộn − nguồn {M - N:+.3f} | phần do máy {(M - N) / (T - N):.0%}")
    h = v(tr, "test_tb_tram") - v(ng, "test_tb_tram")
    print(f"   theo seed (trộn − nguồn): {[f'{x:+.3f}' for x in h]}")
    print("\nTheo trung tâm (test thật của trung tâm đó):")
    print(f"   {'':<14}" + "".join(f"{t:>11}" for t in TT))
    for nhan, x in (("tập trung", tt), ("FedAvg nguồn", ng), ("FedAvg trộn", tr)):
        print(f"   {nhan:<14}" + "".join(f"{np.nanmean(v(x, 'test_' + t)):>11.3f}" for t in TT))
    print("\nLuật: G8b lặp lại nếu trộn − nguồn > 0,03 VÀ dương ở cả 3 seed. HAM10000 (EfficientNet): +0,048, phần do máy 40%.")


if __name__ == "__main__":
    main()
