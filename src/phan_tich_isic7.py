"""
phan_tich_isic7.py — Lượt 25: khoảng tập trung − FedAvg trên Fed-ISIC khi CẢ HAI cùng ngân sách dài (tập trung 60 epoch ≈ FedAvg 27 vòng × 100 bước × 6 trung tâm).
Công thức HAM, bAcc test TB đều 6 trung tâm, 3 seed.
Luật viết trước: "ConvNeXt thiệt ít hơn khi FL" đúng nếu khoảng(EffNet) − khoảng(ConvNeXt) > 0,02 VÀ dương ở cả 3 seed (ghép theo seed).
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
MH = [("EfficientNet-B0", "efficientnet_b0"), ("ConvNeXt-atto", "convnext_atto")]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'mô hình':<17}{'TT 20ep':>9}{'TT 60ep':>9}{'FedAvg 27×100':>15}{'khoảng (60ep)':>15}   khoảng theo seed")
    k = {}
    for nhan, bb in MH:
        t20 = doc(f"isic_tap_trung_cc_{bb}_ham"); t60 = doc(f"isic_tap_trung_cc_{bb}_ham_e60")
        fa = doc(f"isic_fedavg_cc_{bb}_ham_v27b100"); k[bb] = t60 - fa
        print(f"{nhan:<17}{np.nanmean(t20):>9.3f}{np.nanmean(t60):>9.3f}{np.nanmean(fa):>15.3f}{np.nanmean(k[bb]):>15.3f}   {[f'{v:.3f}' for v in k[bb]]}")
    print("\nTheo trung tâm, tập trung 60 epoch (EffNet / ConvNeXt):")
    for t in TT:
        print(f"  {t:<10}{np.nanmean(doc('isic_tap_trung_cc_efficientnet_b0_ham_e60', 'test_' + t)):.3f} / "
              f"{np.nanmean(doc('isic_tap_trung_cc_convnext_atto_ham_e60', 'test_' + t)):.3f}")
    d = k["efficientnet_b0"] - k["convnext_atto"]
    if np.isnan(d).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok = d.mean() > 0.02 and (d > 0).all()
    print(f"\nkhoảng EffNet − khoảng ConvNeXt: {d.mean():+.3f}  theo seed {[f'{v:+.3f}' for v in d]}")
    print("KẾT LUẬN theo luật viết trước:",
          "ĐÚNG — cùng ngân sách dài, ConvNeXt (LayerNorm) thiệt ít hơn khi FL; thiết kế 1 lặp lại trên bộ thứ hai." if ok else
          "KHÔNG được ủng hộ — ConvNeXt hơn ở FedAvg nhưng tập trung cũng hơn tương ứng; khoảng FL không nhỏ hơn rõ.")


if __name__ == "__main__":
    main()
