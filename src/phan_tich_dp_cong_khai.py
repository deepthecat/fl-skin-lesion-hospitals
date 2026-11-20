"""
phan_tich_dp_cong_khai.py — Đêm 27: thiết kế đề xuất. DP lớp cuối (L0) trong FL 4 bệnh viện HAM, xương sống EfficientNet-B0
ĐÓNG BĂNG, so hai nguồn đặc trưng: ImageNet (thiết kế 2 hiện tại) vs học trước trên ảnh da CÔNG KHAI của bệnh viện khác (BCN, ISIC 2019,
đã loại ảnh nghi trùng HAM). Với mỗi ε: lr tốt nhất trên val trong {16, 4, 1}; TB 3 seed.
Luật viết trước: "thiết kế cân bằng" nếu đặc trưng BCN ở ε=3 đạt val ≥ 0,55 (ImageNet: ~0,43).
"""
import json
import warnings

import numpy as np
import pandas as pd

from du_lieu import RA

warnings.filterwarnings("ignore")
MUC = ("inf", "8", "3", "1")
NGUON = {"ImageNet (thiết kế 2)": ({"16": "dp_tuyen_tinh", "4": "dp_tuyen_tinh_lr4", "1": "dp_tuyen_tinh_lr1"}, ""),
         "BCN công khai (đề xuất)": ({"16": "dp_tuyen_tinh_bcn_lr16", "4": "dp_tuyen_tinh_bcn_lr4", "1": "dp_tuyen_tinh_bcn_lr1"}, "_bcn")}


def main():
    hang = []
    for nguon, (thu, nh) in NGUON.items():
        for lr, d in thu.items():
            for m in MUC:
                for s in range(3):
                    f = RA / d / f"dptt_nguon_eps{m}_efficientnet_b0{nh}_txlp1_kt224_s{s}.json"
                    if f.exists():
                        k = json.load(open(f))["ket_qua"]
                        hang.append(dict(nguon=nguon, lr=lr, eps=m, seed=s, val=k["val"]["bacc"], test=k["test"]["bacc"]))
    if not hang:
        raise SystemExit("Chưa có kết quả.")
    b = pd.DataFrame(hang)
    g = b.groupby(["nguon", "eps", "lr"]).agg(val=("val", "mean"), test=("test", "mean"), sd=("val", "std"), n=("val", "count")).reset_index()
    tot = g.loc[g.groupby(["nguon", "eps"]).val.idxmax()]
    pd.set_option("display.width", 200)
    print("FL 4 bệnh viện HAM, DP lớp cuối, EfficientNet-B0 đóng băng — val bAcc (lr tốt nhất cho từng ε, TB 3 seed)\n")
    print(tot.pivot(index="nguon", columns="eps", values="val")[list(MUC)].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTest bAcc tương ứng:\n")
    print(tot.pivot(index="nguon", columns="eps", values="test")[list(MUC)].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nlr được chọn / số seed:\n")
    print(tot.assign(x=tot.lr + " / " + tot.n.astype(str)).pivot(index="nguon", columns="eps", values="x")[list(MUC)].to_string())
    try:
        v3 = tot[(tot.nguon == "BCN công khai (đề xuất)") & (tot.eps == "3")].val.iat[0]
        v3i = tot[(tot.nguon == "ImageNet (thiết kế 2)") & (tot.eps == "3")].val.iat[0]
    except IndexError:
        print("\nTHIẾU lượt ε=3 — chưa kết luận."); return
    print(f"\nε=3: BCN {v3:.3f} vs ImageNet {v3i:.3f} (chênh {v3 - v3i:+.3f}). Tham khảo không DP, học toàn bộ: EfficientNet FedAvg 0,661 · ConvNeXt FedAvg 0,741.")
    print("KẾT LUẬN theo luật viết trước:",
          "ĐẠT — có thiết kế cân bằng (ε=3, val ≥ 0,55)." if v3 >= 0.55 else
          "KHÔNG đạt 0,55 — kết luận đánh đổi đứng, kèm mức mà đòn bẩy 'đặc trưng tốt hơn' kéo được.")


if __name__ == "__main__":
    main()
