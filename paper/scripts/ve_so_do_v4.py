"""ve_so_do_v4.py — bản đơn giản: chỉ tên hộp, không nhãn mục, không dấu đánh số, không chú thích nhỏ trong hộp.
    cd v4 && python ../ve_so_do_v4.py
Ra hinh_paper/: hinh1_pipeline.pdf, hinh2_nevi.pdf, hinh3_thiet_ke.pdf"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from ve_so_do_v2 import MAU, VIEN, duong, hop, khung, luu, mui

CO = 5.6


def h(ax, x, y, w, hh, ten, mau):
    return hop(ax, x, y, w, hh, ten, [], mau, co=CO)


def khung_bv(ax, x0, y0, w, hh, ten):
    for k in (2, 1, 0):
        ax.add_patch(FancyBboxPatch((x0 + 0.04 * k, y0 + 0.04 * k), w, hh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=MAU["khung"], ec="black", lw=VIEN))
    ax.text(x0 + 0.08, y0 + hh - 0.1, ten, ha="left", va="center", fontsize=CO, fontweight="bold")


def hinh1():
    fig, ax = khung(7.0, 1.95)
    d = h(ax, 0.02, 1.38, 0.95, 0.38, "Training images", MAU["du_lieu"])
    s = h(ax, 0.02, 0.8, 0.95, 0.38, "Site split", MAU["du_lieu"])
    mui(ax, (d[4], d[2]), (d[4], s[3]))
    X0, Y0, W, H = 1.17, 0.82, 5.65, 1.0
    khung_bv(ax, X0, Y0, W, H, "Hospital k")
    yc, hh = 1.26, 0.42
    b1 = h(ax, 1.28, yc - hh / 2, 0.72, hh, "Local images", MAU["du_lieu"])
    b2 = h(ax, 2.1, yc - hh / 2, 0.8, hh, "Preprocessing", MAU["du_lieu"])
    b3 = h(ax, 3.0, yc - hh / 2, 0.78, hh, "Augmentation", MAU["du_lieu"])
    bx0, bw, bh = 3.9, 1.0, 0.7
    bb = hop(ax, bx0, yc - bh / 2, bw, bh, None, [], MAU["mang"])
    ax.text(bx0 + bw / 2, yc + bh / 2 - 0.09, "Backbone", ha="center", va="center", fontsize=CO, fontweight="bold")
    n1 = h(ax, bx0 + 0.12, yc - 0.06, bw - 0.34, 0.18, "Block", "white")
    n2 = h(ax, bx0 + 0.12, yc - 0.3, bw - 0.34, 0.16, "Norm", MAU["may_chu"])
    mui(ax, (n1[4], n1[2]), (n1[4], n2[3]))
    ax.text(n1[1] + 0.05, (n1[3] + n2[2]) / 2, "N×", ha="left", va="center", fontsize=CO - 0.6)
    b5 = h(ax, 5.02, yc - hh / 2, 0.68, hh, "Classifier", MAU["mang"])
    b6 = h(ax, 5.8, yc - hh / 2, 0.92, hh, "Local update", MAU["hoc"])
    for a, b in ((b1, b2), (b2, b3), (b3, bb), (bb, b5), (b5, b6)):
        mui(ax, (a[1], yc), (b[0], yc))
    duong(ax, [s[1], 1.1], [s[5], s[5]]); duong(ax, [1.1, 1.1], [s[5], yc]); mui(ax, (1.1, yc), (b1[0], yc))
    sv = h(ax, 5.55, 0.06, 1.2, 0.4, "Server averaging", MAU["may_chu"])
    mui(ax, (b6[4], Y0), (b6[4], sv[3]), "site models", lech=(0.06, 0), ha="left")
    xr = bb[4]
    duong(ax, [sv[0], xr], [0.36, 0.36]); mui(ax, (xr, 0.36), (xr, Y0))
    ax.text(xr + 0.06, 0.6, "global model, next round", ha="left", va="center", fontsize=CO - 0.8, style="italic")
    g = h(ax, 3.0, 0.06, 1.0, 0.24, "Final model", MAU["mang"])
    duong(ax, [sv[0], 4.4], [0.18, 0.18]); mui(ax, (4.4, 0.18), (g[1], 0.18))
    e1 = h(ax, 1.17, 0.3, 1.45, 0.24, "Accuracy", MAU["danh_gia"])
    e2 = h(ax, 1.17, 0.0, 1.45, 0.24, "Membership attack", MAU["rieng_tu"])
    duong(ax, [g[0], 2.8], [0.18, 0.18]); duong(ax, [2.8, 2.8], [e1[5], e2[5]])
    mui(ax, (2.8, e1[5]), (e1[1], e1[5])); mui(ax, (2.8, e2[5]), (e2[1], e2[5]))
    luu(fig, "hinh1_pipeline")


def hinh3():
    fig, ax = khung(7.0, 0.98)
    p1 = h(ax, 0.02, 0.56, 1.0, 0.36, "Public images", MAU["du_lieu"])
    p2 = h(ax, 0.02, 0.0, 1.0, 0.36, "Pretraining", MAU["mang"])
    mui(ax, (p1[4], p1[2]), (p1[4], p2[3]))
    X0, Y0, W, H = 1.25, 0.06, 3.95, 0.82
    khung_bv(ax, X0, Y0, W, H, "Hospital k")
    yc, hh = 0.5, 0.36
    a = h(ax, 1.4, yc - hh / 2, 0.85, hh, "Local images", MAU["du_lieu"])
    b = h(ax, 2.4, yc - hh / 2, 1.05, hh, "Frozen backbone", MAU["mang"])
    c = h(ax, 3.6, yc - hh / 2, 1.45, hh, "Classifier with DP-SGD", MAU["rieng_tu"])
    mui(ax, (a[1], yc), (b[0], yc)); mui(ax, (b[1], yc), (c[0], yc))
    duong(ax, [p2[1], b[4]], [p2[5], p2[5]]); mui(ax, (b[4], p2[5]), (b[4], b[2]))
    sv = h(ax, 5.75, yc - hh / 2, 1.2, hh, "Server averaging", MAU["may_chu"])
    xa = X0 + W + 0.09
    mui(ax, (xa, yc + 0.08), (sv[0], yc + 0.08)); mui(ax, (sv[0], yc - 0.08), (xa, yc - 0.08))
    ax.text((xa + sv[0]) / 2, yc + 0.18, "classifiers", ha="center", va="center", fontsize=CO - 0.8, style="italic")
    luu(fig, "hinh3_thiet_ke")


def hinh2():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42})
    a = np.asarray(Image.open('/home/claude/khoi_phuc/hinh_that/nguon_anh_mau_NV.png').convert("RGB"))
    rows = [(49, 199), (261, 411), (472, 622), (683, 833)]
    cols = [(17, 217), (234, 434), (451, 651), (668, 868), (885, 1085), (1103, 1303)]
    ten = ["molemax, Vienna", "modern, Vienna", "Rosendahl, Queensland", "dias, Vienna"]
    fig, axs = plt.subplots(4, 6, figsize=(3.5, 2.13), gridspec_kw=dict(hspace=0.3, wspace=0.06, left=0.005,
                                                                        right=0.995, top=0.955, bottom=0.005))
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(cols):
            axs[r, c].imshow(a[y0:y1, x0:x1], aspect="auto"); axs[r, c].axis("off")
        axs[r, 0].set_title(ten[r], loc="left", fontsize=5.5, pad=1.5)
    fig.savefig("hinh_paper/hinh2_nevi.pdf", dpi=600); fig.savefig("hinh_paper/hinh2_nevi.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    hinh1(); hinh3(); hinh2()
    print("ok")
