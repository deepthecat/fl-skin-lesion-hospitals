"""
phan_tich_ablation.py — CỔNG B: rút từng thành phần khỏi hai thiết kế cuối, xem thiết kế yếu đi bao nhiêu. Không cần GPU.

    python phan_tich_ablation.py

Thiết kế 1 (độ chính xác): ConvNeXt-atto, học toàn bộ, FedAvg, nguồn thật, P1, ảnh 224, trọng số lớp nghịch đảo (toàn cục), 30 vòng × 1 epoch.
Thiết kế 2 (riêng tư):    EfficientNet-B0 đóng băng (ImageNet) + DP-SGD lớp cuối, lr 4, 30 vòng, P1, mỗi bệnh viện tiêu ε riêng.
Luật viết trước: một thành phần "có lý do để nằm đó" nếu rút/đổi nó làm val bAcc giảm > 0,02 VÀ giảm ở cả 3 seed.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")


def doc(thu, ten):
    v, t = [], []
    for s in range(3):
        f = RA / thu / f"{ten}_s{s}.json"
        k = json.load(open(f)).get("ket_qua") if f.exists() else None
        if k and "val" in k:
            v.append(k["val"]["bacc"]); t.append(k["test"]["bacc"] if "test" in k else np.nan)
        else:
            v.append(np.nan); t.append(np.nan)
    return np.array(v), np.array(t)


def bang(tieu_de, goc, cac, dinh_dang_eps=None):
    print(tieu_de)
    gv, gt = goc[1]
    print(f"   {'cấu hình':<34}{'val TB':>8}{'sd':>7}{'test TB':>9}   chênh val theo seed vs gốc      kết luận")
    print(f"   {goc[0]:<34}{np.nanmean(gv):>8.3f}{np.nanstd(gv):>7.3f}{np.nanmean(gt):>9.3f}   (gốc)")
    for nhan, (v, t) in cac:
        if np.all(np.isnan(v)):
            print(f"   {nhan:<34}{'—':>8}   (chưa có)"); continue
        h = v - gv
        kl = "thành phần CÓ lý do (rút ra thì tệ hơn rõ)" if (np.nanmean(-h) > 0.02 and np.all(-h[~np.isnan(h)] > 0)) else \
             ("thay thế TỐT HƠN gốc" if (np.nanmean(h) > 0.02 and np.all(h[~np.isnan(h)] > 0)) else "trong nhiễu")
        print(f"   {nhan:<34}{np.nanmean(v):>8.3f}{np.nanstd(v):>7.3f}{np.nanmean(t):>9.3f}   "
              f"{['%+.3f' % x if not np.isnan(x) else '—' for x in h]}  {kl}")
    print()


def main():
    cv = "convnext_atto"
    goc1 = ("GỐC: ConvNeXt + FedAvg", doc("lien_ket", f"fedavg_nguon_{cv}_txlp1_kt224"))
    bang("THIẾT KẾ 1 — độ chính xác (FedAvg nguồn thật, không DP)", goc1, [
        ("− học trước ImageNet", doc("lien_ket", f"fedavg_nguon_{cv}_txlp1_khp_kt224")),
        ("− trọng số lớp", doc("lien_ket", f"fedavg_nguon_{cv}_txlp1_tskhong_kt224")),
        ("trọng số lớp tính tại chỗ", doc("lien_ket", f"fedavg_nguon_{cv}_txlp1_cb_kt224")),
        ("2 epoch tại chỗ / vòng", doc("lien_ket", f"fedavg_nguon_{cv}_txlp1_e2_kt224")),
        ("ConvNeXt → EfficientNet-B0", doc("lien_ket", "fedavg_nguon_efficientnet_b0_txlp1_kt224")),
        ("FedAvg → FedProx μ=0,01", doc("lien_ket", f"fedprox_nguon_{cv}_txlp1_kt224")),
        ("P1 → P2 (cân màu)", doc("lien_ket", f"fedavg_nguon_{cv}_txlp2_kt224")),
    ])
    for m in ("8", "3", "1"):
        e = "efficientnet_b0"
        goc2 = (f"GỐC: EffNet-B0 L0, lr 4", doc("dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_{e}_txlp1_kt224"))
        bang(f"THIẾT KẾ 2 — riêng tư, ε = {m} (DP lớp cuối, nguồn thật)", goc2, [
            ("lr 4 → 16 (chọn khi không DP)", doc("dp_tuyen_tinh", f"dptt_nguon_eps{m}_{e}_txlp1_kt224")),
            ("lr 4 → 1", doc("dp_tuyen_tinh_lr1", f"dptt_nguon_eps{m}_{e}_txlp1_kt224")),
            ("30 vòng → 10 vòng", doc("dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_{e}_txlp1_v10_kt224")),
            ("30 vòng → 60 vòng", doc("dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_{e}_txlp1_v60_kt224")),
            ("P1 → P2 (cân màu)", doc("dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_{e}_txlp2_kt224")),
            ("EffNet-B0 → ConvNeXt-atto", doc("dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_convnext_atto_txlp1_kt224")),
            ("EffNet-B0 → ViT-S in21k", doc("dp_tuyen_tinh_lr4", f"dptt_nguon_eps{m}_vit_small_patch16_224.augreg_in21k_ft_in1k_txlp1_kt224")),
            ("(tham chiếu) DP tập trung 1 máy", doc("dp_tuyen_tinh_lr4", f"dptt_mot_may_eps{m}_{e}_txlp1_kt224")),
        ])
    print("Đọc: 'trong nhiễu' = không đủ chứng cứ để nói thành phần đó giúp hay hại (3 seed, ngưỡng 0,02).")
    print("Dòng '(tham chiếu) DP tập trung' cho biết phần thiệt do FL khi đã có DP, không phải ablation.")


if __name__ == "__main__":
    main()
