"""
phan_tich_isic6.py — Lượt 24: ConvNeXt có vượt EfficientNet trên Fed-ISIC khi huấn luyện FL DÀI HƠN (gấp 3 tổng bước)?
FedAvg, công thức HAM, bAcc test TB đều 6 trung tâm, 3 seed. So 3 lịch: 9×100 (đêm 22), 27×33 (đêm 23, cùng tổng bước), 27×100 (gấp 3 tổng bước).
Luật viết trước: "ConvNeXt thắng khi học lâu hơn" đúng nếu ở 27×100, ConvNeXt − EfficientNet > 0,02 VÀ dương ở cả 3 seed.
Lưu ý: 27×100 học nhiều bước hơn mô hình tập trung (20 epoch) → chỉ so HAI KIẾN TRÚC với nhau, không tính khoảng tập trung − FL.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
LICH = [("9 × 100", ""), ("27 × 33", "_v27b33"), ("27 × 100 (×3 bước)", "_v27b100")]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — FedAvg, công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'lịch FL':<22}{'EfficientNet-B0':>17}{'ConvNeXt-atto':>15}{'ConvNeXt − EffNet':>19}   chênh theo seed")
    for nhan, hau in LICH:
        e = doc(f"isic_fedavg_cc_efficientnet_b0_ham{hau}"); c = doc(f"isic_fedavg_cc_convnext_atto_ham{hau}")
        print(f"{nhan:<22}{np.nanmean(e):>17.3f}{np.nanmean(c):>15.3f}{np.nanmean(c - e):>+19.3f}   {[f'{v:+.3f}' for v in c - e]}")
    h = "_v27b100"
    print("\nTheo trung tâm, 27 × 100 (EffNet / ConvNeXt):")
    for t in TT:
        print(f"  {t:<10}{np.nanmean(doc('isic_fedavg_cc_efficientnet_b0_ham' + h, 'test_' + t)):.3f} / "
              f"{np.nanmean(doc('isic_fedavg_cc_convnext_atto_ham' + h, 'test_' + t)):.3f}")
    d = doc("isic_fedavg_cc_convnext_atto_ham" + h) - doc("isic_fedavg_cc_efficientnet_b0_ham" + h)
    if np.isnan(d).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok = d.mean() > 0.02 and (d > 0).all()
    print("\nKẾT LUẬN theo luật viết trước:",
          "ĐÚNG — lợi thế ConvNeXt xuất hiện khi học FL lâu hơn (khớp Siomos TMLR 2025)." if ok else
          "KHÔNG được ủng hộ — học lâu gấp 3 vẫn hoà; lợi thế kiến trúc chỉ có trên HAM ở ngân sách này.")


if __name__ == "__main__":
    main()
