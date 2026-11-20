"""
phan_tich_isic8.py — Lượt 26: phép trộn máy (G8b) trên Fed-ISIC ở CÙNG ngân sách dài với phép so kiến trúc (FedAvg 27 vòng × 100 bước, công thức HAM).
Luật viết trước: G8b lặp lại sạch nếu EfficientNet (BN): trộn − nguồn > 0,03 và dương cả 3 seed; ConvNeXt (LN): |trộn − nguồn| < 0,02.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
MH = [("EfficientNet-B0 (BN)", "efficientnet_b0"), ("ConvNeXt-atto (LN)", "convnext_atto")]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — FedAvg 27×100, công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'mô hình':<22}{'TT 60ep':>9}{'nguồn':>8}{'trộn':>8}{'trộn − nguồn':>14}{'phần do máy':>13}   trộn − nguồn theo seed")
    d = {}
    for nhan, bb in MH:
        t = doc(f"isic_tap_trung_cc_{bb}_ham_e60"); n = doc(f"isic_fedavg_cc_{bb}_ham_v27b100"); m = doc(f"isic_fedavg_cc_{bb}_ham_tron_v27b100")
        d[bb] = m - n; kc = np.nanmean(t - n)
        print(f"{nhan:<22}{np.nanmean(t):>9.3f}{np.nanmean(n):>8.3f}{np.nanmean(m):>8.3f}{np.nanmean(d[bb]):>+14.3f}{np.nanmean(d[bb]) / kc:>12.0%}   {[f'{v:+.3f}' for v in d[bb]]}")
    print("\nTheo trung tâm (nguồn → trộn):")
    for nhan, bb in MH:
        print(f"  {nhan}: " + "  ".join(f"{t} {np.nanmean(doc(f'isic_fedavg_cc_{bb}_ham_v27b100', 'test_' + t)):.2f}→{np.nanmean(doc(f'isic_fedavg_cc_{bb}_ham_tron_v27b100', 'test_' + t)):.2f}" for t in TT))
    e, c = d["efficientnet_b0"], d["convnext_atto"]
    if np.isnan(e).any() or np.isnan(c).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok_e = e.mean() > 0.03 and (e > 0).all(); ok_c = abs(c.mean()) < 0.02
    print(f"\nKẾT LUẬN theo luật viết trước: EfficientNet {'ĐẠT' if ok_e else 'KHÔNG đạt'} (> 0,03, cả 3 seed) · ConvNeXt {'ĐẠT' if ok_c else 'KHÔNG đạt'} (|.| < 0,02)")
    print("→ G8b lặp lại SẠCH trên Fed-ISIC ở ngân sách dài." if ok_e and ok_c else "→ G8b chỉ lặp lại một phần ở ngân sách dài — xem từng vế.")


if __name__ == "__main__":
    main()
