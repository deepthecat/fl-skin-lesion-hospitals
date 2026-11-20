"""
ve_hinh_paper.py — vẽ các biểu đồ cho paper TRỰC TIẾP từ kết quả đã chạy (ketqua/*.json). Không cần GPU, vài giây.

    python ve_hinh_paper.py              # đọc ketqua/*.json, ghi hình + hinh_paper_du_lieu.json
    python ve_hinh_paper.py --tu-json F  # vẽ lại chỉ từ file dữ liệu F (không cần ketqua/)

Ra (ketqua/hinh_paper/, .pdf + .png):
    hinh4_chuan_hoa  — trộn thiết bị thay đổi độ chính xác bao nhiêu, theo kiểu chuẩn hoá
    hinh5_rieng_tu   — đánh đổi riêng tư – độ chính xác (DP lớp cuối, đặc trưng ImageNet và BCN)
    hinh6_duong_hoc  — độ chính xác theo vòng FedAvg
    + hinh_paper_du_lieu.json: mọi con số đã vẽ.
Kiểu hình (học theo Fig. 3 và Fig. 5 của Group 2): chữ không chân mặc định của matplotlib, chữ nhỏ so với vùng vẽ,
khung đủ 4 cạnh, không lưới, không chú số trên cột, vài màu dịu có viền mảnh; mỗi hình chỉ giữ số liệu của một ý.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XANH, CAM, XAM, LUC = "#4C72B0", "#DD8452", "#8C8C8C", "#55A868"        # bảng màu "deep": dịu, in xám vẫn khác độ sáng
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 5, "axes.labelsize": 5.5, "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.labelpad": 3, "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.6,
    "lines.linewidth": 0.8, "savefig.dpi": 600, "pdf.fonttype": 42,
})
RONG = 3.4                                              # inch, vừa một cột IEEE (3.5 in)


def luu(fig, thu, ten):
    for duoi in ("pdf", "png"):
        fig.savefig(thu / f"{ten}.{duoi}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ================================================================ đọc số liệu từ ketqua/
def doc_so_lieu(RA):
    def val(thu, ten):
        f = RA / thu / f"{ten}.json"
        return json.load(open(f))["ket_qua"]["val"]["bacc"] if f.exists() else np.nan

    def tb(xs):
        x = np.array(xs, float)
        return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")

    D = {}
    mh = [("efficientnet_b0", 224, "EfficientNet-B0", "BN"), ("resnet18", 224, "ResNet-18", "BN"),
          ("resnet50.a1h_in1k", 160, "ResNet-50", "BN"), ("resnet50_gn.a1h_in1k", 160, "ResNet-50", "GN"),
          ("convnext_atto", 224, "ConvNeXt-atto", "LN"), ("deit_tiny_patch16_224", 224, "DeiT-tiny", "LN")]
    D["hinh2"] = [{"ten": ten, "chuan": cn,
                   "tap_trung": tb([val("tap_trung", f"{bb}_kt{kt}_nghich_txlp1_s{s}") for s in range(3)]),
                   "nguon": tb([val("lien_ket", f"fedavg_nguon_{bb}_txlp1_kt{kt}_s{s}") for s in range(3)]),
                   "tron": tb([val("lien_ket", f"fedavg_tron_nguon_{bb}_txlp1_kt{kt}_s{s}") for s in range(3)])}
                  for bb, kt, ten, cn in mh]

    D["hinh3"] = {}
    for bb, ten in (("efficientnet_b0", "EfficientNet-B0 (BN)"), ("convnext_atto", "ConvNeXt-atto (LN)")):
        dong = []
        for s in range(3):
            f = RA / "lien_ket" / f"fedavg_nguon_{bb}_txlp1_kt224_s{s}.json"
            if f.exists():
                dong.append([h["val_bacc"] for h in json.load(open(f))["lich_su"] if "val_bacc" in h])
        if dong:
            n = min(map(len, dong))
            D["hinh3"][ten] = [d[:n] for d in dong]

    eps = ["1", "3", "8", "inf"]
    nguon = {"ImageNet features": (["dp_tuyen_tinh", "dp_tuyen_tinh_lr4", "dp_tuyen_tinh_lr1"], ""),
             "Public dermoscopy features (BCN)": (["dp_tuyen_tinh_bcn_lr16", "dp_tuyen_tinh_bcn_lr4", "dp_tuyen_tinh_bcn_lr1"], "_bcn")}
    D["hinh4"] = {"duong": {}, "moc": {}}
    for ten, (thu, nh) in nguon.items():
        D["hinh4"]["duong"][ten] = [float(np.nanmax([tb([val(d, f"dptt_nguon_eps{e}_efficientnet_b0{nh}_txlp1_kt224_s{s}")
                                                         for s in range(3)]) for d in thu])) for e in eps]
    D["hinh4"]["moc"]["ConvNeXt-atto"] = tb([val("lien_ket", f"fedavg_nguon_convnext_atto_txlp1_kt224_s{s}") for s in range(5)])
    D["hinh4"]["moc"]["EfficientNet-B0"] = tb([val("lien_ket", f"fedavg_nguon_efficientnet_b0_txlp1_kt224_s{s}") for s in range(5)])
    return D


# ================================================================ Hình 4: trộn thiết bị thay đổi độ chính xác bao nhiêu
def hinh4(D, thu):
    R = D["hinh2"]
    mau = {"BN": XANH, "GN": XAM, "LN": CAM}
    print("Hình 4 — val bAcc TB 3 seed: FedAvg nguồn / FedAvg trộn / chênh (trộn − nguồn) / phần do thiết bị:")
    y = np.arange(len(R))[::-1]
    fig, ax = plt.subplots(figsize=(RONG, 1.75))
    for yi, r in zip(y, R):
        d = r["tron"] - r["nguon"]
        p = d / (r["tap_trung"] - r["nguon"])
        print(f"  {r['ten']:<16}{r['chuan']}  {r['nguon']:.3f}  {r['tron']:.3f}  {d:+.3f}  {p:+.0%}")
        ax.barh(yi, d, height=0.62, color=mau[r["chuan"]], edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([r["ten"] + (f" ({r['chuan']})" if r["ten"] == "ResNet-50" else "") for r in R])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.055, 0.065); ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04, 0.06])
    ax.set_xlabel("Balanced accuracy, device-mixed minus real sites")
    tay = [plt.Rectangle((0, 0), 1, 1, fc=mau[k], ec="black", lw=0.4) for k in ("BN", "GN", "LN")]
    ax.legend(tay, ["Batch norm", "Group norm", "Layer norm"], loc="upper left", handlelength=1.0, handleheight=0.8)
    luu(fig, thu, "hinh4_chuan_hoa")


# ================================================================ Hình 5: đánh đổi riêng tư – độ chính xác
def hinh5(D, thu):
    fig, ax = plt.subplots(figsize=(RONG, 1.9))
    print("Hình 5 — DP lớp cuối, val bAcc, lr tốt nhất mỗi epsilon, TB 3 seed (eps 1 / 3 / 8 / không DP):")
    kieu = {"ImageNet features": (XAM, "o", "ImageNet features"),
            "Public dermoscopy features (BCN)": (XANH, "s", "Public dermoscopy features")}
    for ten, y in D["hinh4"]["duong"].items():
        mau, mk, nhan = kieu[ten]
        ax.plot(range(4), y, color=mau, marker=mk, ms=3, mec="black", mew=0.35, label=nhan)
        print(f"  {ten:<34}" + "  ".join(f"{a:.3f}" for a in y))
    moc = D["hinh4"]["moc"]["ConvNeXt-atto"]
    ax.axhline(moc, color="black", lw=0.6, ls="--")
    ax.text(3.15, moc + 0.012, "Best federated model without DP", ha="right", va="bottom", fontsize=4.6)
    print(f"  mốc: ConvNeXt-atto FedAvg không DP {moc:.3f} (TB 5 seed)")
    ax.set_xticks(range(4)); ax.set_xticklabels(["1", "3", "8", "no DP"]); ax.set_xlim(-0.25, 3.25)
    ax.set_xlabel("Privacy budget ε of each site"); ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.28, 0.80)
    ax.legend(loc="lower right")
    luu(fig, thu, "hinh5_rieng_tu")


# ================================================================ Hình 6: độ chính xác theo vòng
def hinh6(D, thu):
    fig, ax = plt.subplots(figsize=(RONG, 1.75))
    print("Hình 6 — val bAcc theo vòng, FedAvg nguồn thật, TB 3 seed (vòng 10 / vòng cuối / vòng tốt nhất):")
    mau = {"EfficientNet-B0 (BN)": XANH, "ConvNeXt-atto (LN)": CAM}
    for ten, dong in D["hinh3"].items():
        A = np.array(dong); v = np.arange(1, A.shape[1] + 1); m = A.mean(0)
        ax.fill_between(v, A.min(0), A.max(0), color=mau[ten], alpha=0.18, linewidth=0)
        ax.plot(v, m, color=mau[ten], label=ten)
        print(f"  {ten:<22} {m[9]:.3f} / {m[-1]:.3f} / vòng {int(np.argmax(m)) + 1} ({m.max():.3f})")
    ax.set_xlim(1, 30); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Communication round"); ax.set_ylabel("Balanced accuracy")
    ax.legend(loc="lower right")
    luu(fig, thu, "hinh6_duong_hoc")


if __name__ == "__main__":
    if "--tu-json" in sys.argv:
        f = Path(sys.argv[sys.argv.index("--tu-json") + 1])
        D = json.load(open(f)); RA_HINH = f.parent / "hinh_paper"
    else:
        from du_lieu import RA
        D = doc_so_lieu(RA); RA_HINH = RA / "hinh_paper"
    RA_HINH.mkdir(parents=True, exist_ok=True)
    json.dump(D, open(RA_HINH / "hinh_paper_du_lieu.json", "w"), indent=1, ensure_ascii=False)
    for h in (hinh4, hinh5, hinh6):
        try:
            h(D, RA_HINH)
        except Exception as e:                        # một hình lỗi không chặn hình khác
            print(f"LỖI ở {h.__name__}: {e}")
    print(f"\nĐã ghi vào {RA_HINH}")
