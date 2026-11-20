"""
phan_tich_isic4.py — Thiết kế 1 trên bộ dữ liệu THỨ HAI, lần 2: CÙNG công thức HAM10000 cho cả hai mô hình. Fed-ISIC2019 (6 trung tâm, có BCN và MSK). Không cần GPU.

    python phan_tich_isic4.py

So ConvNeXt-atto (LayerNorm) với EfficientNet-B0 (BatchNorm), CÙNG thư viện timm, CÙNG giao thức:
tập trung (AdamW 3e-4, khởi động + giảm cos, 20 epoch) và FedAvg (AdamW 3e-4 giảm cos theo vòng, 100 bước/vòng, 9 vòng như FLamby). Độ đo: bAcc
trung bình đều 6 trung tâm (cách FLamby báo). Kèm EfficientNet-B0 bản FLamby (efficientnet_pytorch, FedAvg SGD) đã có.
Kiểm trước: ConvNeXt tập trung phải học được (> 0,70), nếu không phép so vẫn không hợp lệ.
Dự đoán (viết trước): khi cả hai đều học được, khoảng tập trung − FedAvg của ConvNeXt NHỎ HƠN của EfficientNet.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]


def doc(ten):
    r = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        r.append(json.load(open(f))["ket_qua"] if f.exists() else None)
    return r


def tb(rs, k):
    x = np.array([r[k] if r else np.nan for r in rs], float)
    return np.nanmean(x), x


def main():
    cau = [("EffNet-B0 (FLamby gốc)", "isic_tap_trung_cc", "isic_fedavg_cc"),
           ("EffNet-B0 (timm, công thức HAM)", "isic_tap_trung_cc_efficientnet_b0_ham", "isic_fedavg_cc_efficientnet_b0_ham"),
           ("ConvNeXt-atto (timm, công thức HAM)", "isic_tap_trung_cc_convnext_atto_ham", "isic_fedavg_cc_convnext_atto_ham")]
    print("Fed-ISIC2019 — bAcc test, TRUNG BÌNH ĐỀU 6 trung tâm (3 seed)\n")
    print(f"{'mô hình':<36}{'chuẩn hoá':<11}{'tập trung':>10}{'FedAvg':>8}{'khoảng':>8}   FedAvg theo seed")
    kq = {}
    for nhan, a, b in cau:
        ta, _ = tb(doc(a), "test_tb_tram"); fb, xb = tb(doc(b), "test_tb_tram")
        kq[nhan] = (doc(a), doc(b))
        cn = "LayerNorm" if "ConvNeXt" in nhan else "BatchNorm"
        ghi = " (FedAvg SGD 0,01)" if "gốc" in nhan else " (FedAvg AdamW 3e-4 cos)"
        print(f"{nhan:<36}{cn:<11}{ta:>10.3f}{fb:>8.3f}{ta - fb:>8.3f}   {[f'{v:.3f}' for v in xb]}{ghi}")
    print("\nTheo trung tâm (FedAvg; trong ngoặc: tập trung):")
    print(f"{'mô hình':<36}" + "".join(f"{t:>14}" for t in TT))
    for nhan, (ra, rb) in kq.items():
        print(f"{nhan:<36}" + "".join(f"{tb(rb, 'test_' + t)[0]:>7.3f}({tb(ra, 'test_' + t)[0]:.3f})" for t in TT))
    print("\nĐọc: so hai dòng timm với nhau (cùng thư viện, cùng giao thức). 'khoảng' nhỏ hơn = FL ít thiệt hơn."
          "\nGiới hạn: công thức HAM không phải giao thức gốc FLamby; dùng để so HAI KIẾN TRÚC với nhau, không để so số tuyệt đối với FLamby.")


if __name__ == "__main__":
    main()
