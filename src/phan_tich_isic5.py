"""
phan_tich_isic5.py — Đêm 23: ConvNeXt thua trên Fed-ISIC vì ÍT VÒNG (giải thích B) hay không?
Cùng công thức HAM, cùng TỔNG số bước: 9 vòng × 100 bước (đêm 22) vs 27 vòng × 33 bước (đêm 23). bAcc test, TB đều 6 trung tâm, 3 seed.
Luật viết trước: B đúng nếu ở 27×33, ConvNeXt − EfficientNet > 0,02 VÀ dương ở cả 3 seed.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — FedAvg, công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'lịch FL':<14}{'EfficientNet-B0':>17}{'ConvNeXt-atto':>15}{'ConvNeXt − EffNet':>19}   chênh theo seed")
    kq = {}
    for nhan, hau in [("9 × 100", ""), ("27 × 33", "_v27b33")]:
        e = doc(f"isic_fedavg_cc_efficientnet_b0_ham{hau}"); c = doc(f"isic_fedavg_cc_convnext_atto_ham{hau}")
        kq[nhan] = (e, c)
        print(f"{nhan:<14}{np.nanmean(e):>17.3f}{np.nanmean(c):>15.3f}{np.nanmean(c - e):>+19.3f}   {[f'{v:+.3f}' for v in c - e]}")
    e, c = kq["27 × 33"]; d = c - e
    print("\nTheo trung tâm, 27 × 33 (EffNet / ConvNeXt):")
    for t in TT:
        print(f"  {t:<10}{np.nanmean(doc('isic_fedavg_cc_efficientnet_b0_ham_v27b33', 'test_' + t)):.3f} / "
              f"{np.nanmean(doc('isic_fedavg_cc_convnext_atto_ham_v27b33', 'test_' + t)):.3f}")
    if np.isnan(d).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok = np.nanmean(d) > 0.02 and (d > 0).all()
    print("\nKẾT LUẬN theo luật viết trước:",
          "B ĐÚNG — ConvNeXt thua ở đêm 22 vì ít vòng; lợi thế LayerNorm lặp lại khi đủ vòng." if ok else
          "B KHÔNG được ủng hộ — lợi thế kiến trúc chỉ có trên HAM; cơ chế G8b vẫn lặp lại (đêm 21).")
    print("Tổng tập trung (đêm 22, để tính khoảng): EffNet 0,738 · ConvNeXt 0,733.")


if __name__ == "__main__":
    main()
