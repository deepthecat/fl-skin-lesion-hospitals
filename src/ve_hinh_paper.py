"""
ve_hinh_paper.py — vẽ hình cho paper TRỰC TIẾP từ kết quả đã chạy (ketqua/*.json). Không cần GPU, vài giây.

    python ve_hinh_paper.py              # đọc ketqua/*.json, ghi hình + hinh_paper_du_lieu.json
    python ve_hinh_paper.py --tu-json F  # vẽ lại chỉ từ file dữ liệu F (không cần ketqua/)

Ra: ketqua/hinh_paper/hinh2_thiet_bi, hinh3_duong_hoc, hinh4_rieng_tu (.pdf + .png)
    + hinh_paper_du_lieu.json: mọi con số đã vẽ (để đối chiếu bảng và để chỉnh hình mà không cần chạy lại).
Kiểu hình: đơn sắc (đen / xám), một cỡ chữ 7 pt, chữ có chân như thân bài IEEE, khung mảnh, không tiêu đề
(thông điệp nằm ở chú thích dưới hình), rộng 3.5 in = đúng một cột IEEE nên không bị co chữ khi chèn.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

DEN, XAM_DAM, XAM, XAM_NHAT = "#000000", "#555555", "#999999", "#d9d9d9"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6", "grid.linewidth": 0.4, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.6, "lines.linewidth": 1.0,
    "hatch.linewidth": 0.4, "savefig.dpi": 600, "pdf.fonttype": 42,
})
RONG = 3.5                                              # inch, một cột IEEE


def luu(fig, thu, ten):
    for duoi in ("pdf", "png"):
        fig.savefig(thu / f"{ten}.{duoi}", bbox_inches="tight", pad_inches=0.02)
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


# ================================================================ Hình 2: phần khoảng cách do thiết bị
def hinh2(D, thu):
    R = D["hinh2"]
    print("Hình 2 — val bAcc, TB 3 seed (tập trung / FedAvg nguồn / FedAvg trộn / phần do thiết bị):")
    for r in R:
        p = (r["tron"] - r["nguon"]) / (r["tap_trung"] - r["nguon"])
        print(f"  {r['ten']:<16}{r['chuan']}  {r['tap_trung']:.3f}  {r['nguon']:.3f}  {r['tron']:.3f}  {p:+.0%}")
    x = np.arange(len(R)); w = 0.26
    fig, ax = plt.subplots(figsize=(RONG, 2.0))
    kieu = dict(edgecolor=DEN, linewidth=0.4)
    ax.bar(x - w, [r["tap_trung"] for r in R], w, color="white", label="Centralized", **kieu)
    ax.bar(x, [r["nguon"] for r in R], w, color=XAM_DAM, label="FedAvg, real sites", **kieu)
    ax.bar(x + w, [r["tron"] for r in R], w, color=XAM_NHAT, label="FedAvg, device-mixed", **kieu)
    for i, r in enumerate(R):
        p = (r["tron"] - r["nguon"]) / (r["tap_trung"] - r["nguon"])
        nhan = f"{p:.0%}" if p >= 0.1 else ("≈0" if p > -0.1 else "<0")
        ax.text(x[i], r["tap_trung"] + 0.006, nhan, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([r["ten"].replace("-", "-\n", 1) + f" ({r['chuan']})" for r in R], fontsize=6.5, linespacing=1.1)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(R) - 0.45)
    ax.set_ylim(0.60, 0.86); ax.set_yticks(np.arange(0.60, 0.861, 0.05))
    ax.set_ylabel("Balanced accuracy")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.2, borderaxespad=0.2)
    luu(fig, thu, "hinh2_thiet_bi")


# ================================================================ Hình 3: đường học theo vòng
def hinh3(D, thu):
    fig, ax = plt.subplots(figsize=(RONG, 1.9))
    print("Hình 3 — val bAcc theo vòng, FedAvg nguồn thật, TB 3 seed (vòng 10 / vòng cuối / vòng tốt nhất):")
    kieu = {"EfficientNet-B0 (BN)": (XAM_DAM, "--"), "ConvNeXt-atto (LN)": (DEN, "-")}
    for ten, dong in D["hinh3"].items():
        A = np.array(dong); v = np.arange(1, A.shape[1] + 1); m = A.mean(0)
        mau, ls = kieu[ten]
        ax.fill_between(v, A.min(0), A.max(0), color=XAM_NHAT, alpha=0.6, linewidth=0)
        ax.plot(v, m, color=mau, ls=ls, label=ten)
        print(f"  {ten:<22} {m[9]:.3f} / {m[-1]:.3f} / vòng {int(np.argmax(m)) + 1} ({m.max():.3f})")
    ax.set_xlim(1, max(len(d[0]) for d in D["hinh3"].values()))
    ax.set_xlabel("Communication round"); ax.set_ylabel("Balanced accuracy")
    ax.legend(loc="lower right")
    luu(fig, thu, "hinh3_duong_hoc")


# ================================================================ Hình 4: đánh đổi riêng tư – độ chính xác
def hinh4(D, thu):
    fig, ax = plt.subplots(figsize=(RONG, 2.0))
    print("Hình 4 — DP lớp cuối, val bAcc, lr tốt nhất mỗi epsilon, TB 3 seed (eps 1 / 3 / 8 / không DP):")
    kieu = {"ImageNet features": (XAM_DAM, "o", "--", "white"), "Public dermoscopy features (BCN)": (DEN, "s", "-", DEN)}
    for ten, y in D["hinh4"]["duong"].items():
        mau, mk, ls, ruot = kieu[ten]
        ax.plot(range(4), y, color=mau, ls=ls, marker=mk, ms=3.2, mfc=ruot, mec=mau, mew=0.7, label=ten)
        print(f"  {ten:<34}" + "  ".join(f"{a:.3f}" for a in y))
    for ten, gt in D["hinh4"]["moc"].items():
        ax.axhline(gt, color=XAM, lw=0.6, ls=":")
        ax.text(-0.15, gt + 0.006, f"{ten} FedAvg, no DP", va="bottom", ha="left", fontsize=6.5, color=XAM_DAM)
    print("  mốc FedAvg không DP (TB 5 seed): " + " · ".join(f"{k} {v:.3f}" for k, v in D["hinh4"]["moc"].items()))
    ax.set_xticks(range(4)); ax.set_xticklabels(["1", "3", "8", "no DP"]); ax.set_xlim(-0.2, 3.2)
    ax.set_xlabel("Privacy budget $\\varepsilon$ per site"); ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.30, 0.80); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right")
    luu(fig, thu, "hinh4_rieng_tu")


if __name__ == "__main__":
    if "--tu-json" in sys.argv:
        f = Path(sys.argv[sys.argv.index("--tu-json") + 1])
        D = json.load(open(f)); RA_HINH = f.parent / "hinh_paper"
    else:
        from du_lieu import RA
        D = doc_so_lieu(RA); RA_HINH = RA / "hinh_paper"
    RA_HINH.mkdir(parents=True, exist_ok=True)
    json.dump(D, open(RA_HINH / "hinh_paper_du_lieu.json", "w"), indent=1, ensure_ascii=False)
    for h in (hinh2, hinh3, hinh4):
        try:
            h(D, RA_HINH)
        except Exception as e:                        # một hình lỗi không chặn hình khác
            print(f"LỖI ở {h.__name__}: {e}")
    print(f"\nĐã ghi vào {RA_HINH}")
