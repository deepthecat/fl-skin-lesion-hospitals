"""
phan_tich_dac_trung.py — đặc trưng ĐÓNG BĂNG nào tốt nhất cho DP lớp cuối (L0)? Không cần GPU.

    python phan_tich_dac_trung.py

Vì sao: đêm 16–17 cho thấy ở quy mô HAM10000 (4 bệnh viện, 8,6k ảnh) chỉ DP lớp cuối là còn dùng được. Khi xương sống
đóng băng, đổi sang xương sống TO HƠN không làm tốn thêm ε (chỉ lớp cuối chịu nhiễu) — nên chất lượng đặc trưng
có sẵn là núm vặn chính của trục riêng tư (ý của Tramèr & Boneh, ICLR 2021: DP cần đặc trưng tốt hơn).
Với mỗi xương sống: lấy lr tốt nhất trên val trong {16 (tự chọn khi không DP), 4, 1} cho TỪNG mức ε, báo val/test.
Việc chọn lr theo ε dùng val (không phải dữ liệu huấn luyện) — ghi rõ trong bài.
"""
import json
import warnings

import numpy as np
import pandas as pd

from du_lieu import RA

warnings.filterwarnings("ignore")
XUONG = ["efficientnet_b0", "convnext_atto", "resnet18", "deit_tiny_patch16_224", "efficientnet_b3",
         "resnet50.a1h_in1k", "vit_small_patch16_224.augreg_in21k_ft_in1k", "convnext_tiny.fb_in22k_ft_in1k"]
THU = {"16 (tự chọn)": "dp_tuyen_tinh", "4": "dp_tuyen_tinh_lr4", "1": "dp_tuyen_tinh_lr1"}
MUC = ("inf", "8", "3", "1")


def main():
    hang = []
    for bb in XUONG:
        for lr, thu in THU.items():
            for m in MUC:
                for s in range(3):
                    f = RA / thu / f"dptt_nguon_eps{m}_{bb}_txlp1_kt224_s{s}.json"
                    if f.exists():
                        k = json.load(open(f))["ket_qua"]
                        hang.append(dict(xuong=bb, lr=lr, eps=m, seed=s, val=k["val"]["bacc"], test=k["test"]["bacc"]))
    if not hang:
        raise SystemExit("Chưa có kết quả.")
    b = pd.DataFrame(hang)
    g = b.groupby(["xuong", "eps", "lr"]).agg(val=("val", "mean"), test=("test", "mean"), n=("val", "count")).reset_index()
    tot = g.loc[g.groupby(["xuong", "eps"]).val.idxmax()]
    bang = tot.pivot(index="xuong", columns="eps", values="val")[list(MUC)]
    bang_t = tot.pivot(index="xuong", columns="eps", values="test")[list(MUC)]
    bang = bang.sort_values("3", ascending=False)
    pd.set_option("display.width", 220)
    print("Val bAcc, DP lớp cuối, lr tốt nhất cho từng ε (TB 3 seed), xếp theo ε=3:\n")
    print(bang.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTest bAcc tương ứng:\n")
    print(bang_t.loc[bang.index].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nlr được chọn:\n")
    print(tot.pivot(index="xuong", columns="eps", values="lr").loc[bang.index][list(MUC)].to_string())
    goc = bang.loc["efficientnet_b0", "3"] if "efficientnet_b0" in bang.index else np.nan
    print(f"\nMốc hiện tại (EfficientNet-B0, ε=3): {goc:.3f}. Luật viết trước: xương sống mới chỉ thay mốc nếu hơn > 0,02 ở ε=3"
          "\nVÀ không kém ở ε=8/ε=1. Học toàn bộ không DP (tham khảo): ConvNeXt-atto 0,737, EfficientNet-B0 0,659.")


if __name__ == "__main__":
    main()
