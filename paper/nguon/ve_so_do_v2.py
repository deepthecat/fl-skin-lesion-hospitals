"""
ve_so_do_v2.py — hai sơ đồ cho paper, vẽ đúng theo code thật (huan_luyen_lien_ket.py, du_lieu.py, dp_tuyen_tinh.py).

    python ve_so_do_v2.py

Ra (hinh_paper/, .pdf + .png):
    hinh1_pipeline   — một lượt học liên kết từ ảnh đầu vào tới đánh giá; số 1–4 đánh dấu chỗ bài can thiệp/đo
    hinh3_thiet_ke   — thiết kế đề xuất: đặc trưng BCN công khai, đóng băng, DP-SGD trên lớp tuyến tính
Kiểu (học từ Hình 1–2 của Transformer): đọc một chiều, tên hộp trùng tên mục trong bài, màu nhạt mã hoá loại khối
(dữ liệu / mạng / huấn luyện / máy chủ / đánh giá / riêng tư), chữ tối thiểu.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "savefig.dpi": 600, "pdf.fonttype": 42})
RA = Path("hinh_paper"); RA.mkdir(exist_ok=True)
CO_TD, CO_CHU = 5.4, 4.6
VIEN, NET = 0.6, 0.55
MAU = dict(du_lieu="#F7E6D6", mang="#DDE9F6", hoc="#E9E4F3", may_chu="#FBF1CF", danh_gia="#E1EFDA",
           rieng_tu="#F6DDE0", khung="#F3F3F3")


def hop(ax, x0, y0, w, h, td, dong=(), mau="white", dam=True, co=None):
    """Hộp góc trái dưới (x0, y0). Trả về (trái, phải, dưới, trên, tâm x, tâm y)."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=mau, ec="black", lw=VIEN))
    buoc = 0.088
    n = len(dong) + (1 if td else 0)
    y = y0 + h / 2 + (n - 1) * buoc / 2
    if td:
        ax.text(x0 + w / 2, y, td, ha="center", va="center", fontsize=co or CO_TD,
                fontweight="bold" if dam else "normal")
        y -= buoc + 0.008
    for d in dong:
        ax.text(x0 + w / 2, y, d, ha="center", va="center", fontsize=CO_CHU)
        y -= buoc
    return x0, x0 + w, y0, y0 + h, x0 + w / 2, y0 + h / 2


def mui(ax, p1, p2, nhan=None, lech=(0, 0.055), ha="center", net="-"):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=NET, color="black",
                                                      mutation_scale=5.5, shrinkA=0, shrinkB=0, ls=net))
    if nhan:
        ax.text((p1[0] + p2[0]) / 2 + lech[0], (p1[1] + p2[1]) / 2 + lech[1], nhan, ha=ha, va="center",
                fontsize=CO_CHU, style="italic")


def duong(ax, xs, ys):
    ax.plot(xs, ys, color="black", lw=NET, solid_capstyle="butt")


def so(ax, x, y, n):
    """Dấu đánh số: vòng tròn rỗng, số đen (khớp \\textcircled trong chú thích)."""
    ax.add_patch(Circle((x, y), 0.055, fc="white", ec="black", lw=0.6, zorder=5))
    ax.text(x, y - 0.002, str(n), ha="center", va="center", fontsize=4.6, fontweight="bold", zorder=6)


def khung_vien(ax, x0, y0, w, h, tieu_de, ghi):
    for k in (2, 1, 0):
        ax.add_patch(FancyBboxPatch((x0 + 0.04 * k, y0 + 0.04 * k), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=MAU["khung"], ec="black", lw=VIEN))
    ax.text(x0 + 0.07, y0 + h - 0.09, tieu_de, ha="left", va="center", fontsize=CO_TD, fontweight="bold")
    ax.text(x0 + w - 0.07, y0 + h - 0.09, ghi, ha="right", va="center", fontsize=CO_CHU, style="italic")


def khung(rong, cao):
    fig = plt.figure(figsize=(rong, cao))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, rong); ax.set_ylim(0, cao); ax.axis("off")
    return fig, ax


def luu(fig, ten):
    for duoi in ("pdf", "png"):
        fig.savefig(RA / f"{ten}.{duoi}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ===================================================================== Hình 1: pipeline một lượt FL
def hinh1():
    fig, ax = khung(7.0, 2.35)
    # ---- dữ liệu và cách chia site (trái)
    d = hop(ax, 0.02, 1.62, 1.0, 0.5, "HAM10000 train", ["8,583 images, 7 classes", "split by lesion"], MAU["du_lieu"])
    s = hop(ax, 0.02, 0.72, 1.0, 0.66, "Site split", ["real: one site per source", "device-mixed: same class", "counts, images drawn", "from all four sources"], MAU["du_lieu"])
    mui(ax, (d[4], d[2]), (d[4], s[3]))
    so(ax, s[0] + 0.07, s[3] - 0.07, 1)

    # ---- bệnh viện k (khung xám, 4 lớp chồng)
    X0, Y0, W, H = 1.2, 1.12, 5.62, 1.1
    khung_vien(ax, X0, Y0, W, H, "Hospital k = 1, ..., 4", "images never leave the site")
    yc, h = 1.5, 0.5
    b1 = hop(ax, 1.3, yc - h / 2, 0.66, h, "Local images", ["$n_k$ images", "of site k"], MAU["du_lieu"])
    b2 = hop(ax, 2.06, yc - h / 2, 0.86, h, "Preprocessing", ["P1: short side 224,", "centre crop;", "P2: + Shades of Gray"], MAU["du_lieu"])
    b3 = hop(ax, 3.02, yc - h / 2, 0.74, h, "Augmentation", ["random flips,", "90° rotations"], MAU["du_lieu"])
    b4 = hop(ax, 3.86, yc - h / 2, 1.12, h, "Backbone", ["ImageNet weights;", "blocks of Conv/Attention", "+ Norm: BN | GN | LN"], MAU["mang"])
    b5 = hop(ax, 5.08, yc - h / 2, 0.66, h, "Classifier", ["linear,", "7 classes"], MAU["mang"])
    b6 = hop(ax, 5.84, yc - h / 2, 0.9, h, "Local update", ["weighted cross-entropy,", "AdamW, 1 epoch"], MAU["hoc"])
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)):
        mui(ax, (a[1], yc), (b[0], yc))
    # site split -> bệnh viện
    duong(ax, [s[1], 1.12], [s[5], s[5]]); duong(ax, [1.12, 1.12], [s[5], yc]); mui(ax, (1.12, yc), (b1[0], yc))

    # ---- máy chủ, mô hình chung, đánh giá (hàng dưới, đọc từ phải sang trái)
    sv = hop(ax, 5.22, 0.08, 1.52, 0.66, "Server: FedAvg", ["weighted average (Eq. 1) of all", "weights, incl. BN running mean/var", "(FedBN: BN layers stay at the site)"], MAU["may_chu"])
    xs = b6[4]
    mui(ax, (xs, Y0), (xs, sv[3]), "site models", lech=(0.06, 0), ha="left")
    so(ax, xs - 0.1, (Y0 + sv[3]) / 2, 3)
    so(ax, sv[0] + 0.07, sv[3] - 0.07, 2)
    # vòng lặp: mô hình chung quay lại backbone
    xr = b4[4]
    duong(ax, [sv[0], xr], [0.55, 0.55]); mui(ax, (xr, 0.55), (xr, Y0))
    ax.text(xr - 0.05, 0.9, "global model, next round (30 rounds)", ha="right", va="center", fontsize=CO_CHU, style="italic")
    g = hop(ax, 3.0, 0.08, 1.1, 0.4, "Final global model", ["best round on validation"], MAU["mang"])
    mui(ax, (sv[0], 0.28), (g[1], 0.28))
    e1 = hop(ax, 1.2, 0.46, 1.6, 0.36, "Accuracy", ["balanced accuracy (val, test), device share"], MAU["danh_gia"])
    e2 = hop(ax, 1.2, 0.02, 1.6, 0.36, "Membership attack", ["loss threshold and RMIA"], MAU["rieng_tu"])
    duong(ax, [g[0], 2.9], [0.28, 0.28]); duong(ax, [2.9, 2.9], [e1[5], e2[5]])
    mui(ax, (2.9, e1[5]), (e1[1], e1[5])); mui(ax, (2.9, e2[5]), (e2[1], e2[5]))
    so(ax, e2[0] + 0.07, e2[3] - 0.07, 4)
    luu(fig, "hinh1_pipeline")


# ===================================================================== Hình 3: thiết kế đề xuất (DP)
def hinh3():
    fig, ax = khung(7.0, 1.75)
    p1 = hop(ax, 0.02, 1.1, 1.05, 0.5, "Public data", ["BCN training images", "(ISIC 2019), HAM10000", "duplicates removed"], MAU["du_lieu"])
    p2 = hop(ax, 0.02, 0.36, 1.05, 0.46, "Pretraining", ["EfficientNet-B0,", "centralized, 20 epochs"], MAU["mang"])
    mui(ax, (p1[4], p1[2]), (p1[4], p2[3]))

    X0, Y0, W, H = 1.25, 0.08, 4.05, 1.5
    khung_vien(ax, X0, Y0, W, H, "Hospital k = 1, ..., 4", "images never leave the site")
    yc, h = 1.02, 0.5
    h1 = hop(ax, 1.35, yc - h / 2, 0.66, h, "Local images", ["$n_k$ images,", "P1"], MAU["du_lieu"])
    h2 = hop(ax, 2.12, yc - h / 2, 0.9, h, "Frozen backbone", ["eval mode,", "BN statistics fixed"], MAU["mang"])
    h3 = hop(ax, 3.13, yc - h / 2, 0.82, h, "Features", ["1280-d, length 1,", "computed once"], MAU["mang"])
    h4 = hop(ax, 4.06, yc - h / 2, 1.13, h, "Linear classifier", ["W, b: 7 classes,", "trained with DP-SGD"], MAU["rieng_tu"])
    for a, b in ((h1, h2), (h2, h3), (h3, h4)):
        mui(ax, (a[1], yc), (b[0], yc))
    # trọng số học trước vào backbone
    duong(ax, [p2[1], h2[4]], [p2[5], p2[5]]); mui(ax, (h2[4], p2[5]), (h2[4], h2[2]))
    ax.text((p2[1] + X0) / 2 + 0.05, p2[5] + 0.055, "weights", ha="center", va="center", fontsize=CO_CHU, style="italic")
    # các bước DP-SGD (dưới bộ phân loại)
    st = hop(ax, 3.13, 0.17, 2.06, 0.36, None, ["each step: Poisson sample (expected batch 256) → clip each", "image's gradient to norm 1 → add Gaussian noise $\\sigma_k$"], "white")
    duong(ax, [h4[4], h4[4]], [h4[2], st[3]])
    ax.text(st[0] - 0.05, st[5], "$\\sigma_k$ from $\\varepsilon$, $\\delta = 10^{-5}$\nand site size", ha="right", va="center", fontsize=CO_CHU, style="italic")

    sv = hop(ax, 5.8, 0.82, 1.18, 0.46, "Server", ["weighted average of W, b", "(30 rounds)"], MAU["may_chu"])
    xa = X0 + W + 0.09
    mui(ax, (xa, sv[5] + 0.09), (sv[0], sv[5] + 0.09)); mui(ax, (sv[0], sv[5] - 0.09), (xa, sv[5] - 0.09))
    ax.text((xa + sv[0]) / 2, sv[5] + 0.17, "$W_k, b_k$", ha="center", va="center", fontsize=CO_CHU, style="italic")
    ax.text((xa + sv[0]) / 2, sv[5] - 0.17, "global", ha="center", va="center", fontsize=CO_CHU, style="italic")
    o = hop(ax, 5.8, 0.1, 1.18, 0.5, "Output", ["frozen backbone + averaged", "classifier; each site spends", "its own ε ∈ {8, 3, 1}"], MAU["danh_gia"])
    mui(ax, (sv[4], sv[2]), (sv[4], o[3]))
    luu(fig, "hinh3_thiet_ke")


# ===================================================================== Hình 1 v3: khung chung, ô có nhiều biến thể ghi tên chung
def the(ax, b, muc):
    """Nhãn mục (vd. III-C) nhỏ, xám, góc phải dưới của hộp b — chỉ cho ô có nhiều biến thể."""
    ax.text(b[1] - 0.04, b[2] + 0.045, muc, ha="right", va="center", fontsize=3.9, color="#555555")


def hinh1_v3():
    fig, ax = khung(7.0, 2.1)
    d = hop(ax, 0.02, 1.5, 1.0, 0.46, "Training images", [], MAU["du_lieu"]); the(ax, d, "III-B")
    s = hop(ax, 0.02, 0.86, 1.0, 0.46, "Site split", [], MAU["du_lieu"]); the(ax, s, "III-B")
    mui(ax, (d[4], d[2]), (d[4], s[3]))
    so(ax, s[0] + 0.07, s[3] - 0.07, 1)

    X0, Y0, W, H = 1.2, 0.92, 5.62, 1.08
    khung_vien(ax, X0, Y0, W, H, "Hospital k = 1, ..., K", "images never leave the site")
    yc, h = 1.4, 0.5
    b1 = hop(ax, 1.3, yc - h / 2, 0.6, h, "Local images", ["of site k"], MAU["du_lieu"])
    b2 = hop(ax, 2.0, yc - h / 2, 0.72, h, "Preprocessing", [], MAU["du_lieu"]); the(ax, b2, "III-C")
    b3 = hop(ax, 2.82, yc - h / 2, 0.74, h, "Augmentation", [], MAU["du_lieu"]); the(ax, b3, "III-C")
    # backbone: khung lớn, bên trong khối + chuẩn hoá, lặp N lần
    bx0, bw, bh = 3.68, 1.12, 0.8
    bb = hop(ax, bx0, yc - bh / 2, bw, bh, None, [], MAU["mang"])
    ax.text(bx0 + bw / 2, yc + bh / 2 - 0.08, "Backbone", ha="center", va="center", fontsize=CO_TD, fontweight="bold")
    n1 = hop(ax, bx0 + 0.14, yc - 0.06, bw - 0.36, 0.2, "Block", [], "white", co=CO_CHU + 0.3)
    n2 = hop(ax, bx0 + 0.14, yc - 0.33, bw - 0.36, 0.18, "Norm", [], MAU["may_chu"], co=CO_CHU + 0.3)
    mui(ax, (n1[4], n1[2]), (n1[4], n2[3]))
    ax.text(n1[1] + 0.05, (n1[3] + n2[2]) / 2, "N×", ha="left", va="center", fontsize=CO_CHU)
    the(ax, bb, "III-D")
    b5 = hop(ax, 4.92, yc - h / 2, 0.66, h, "Classifier", ["linear layer"], MAU["mang"])
    b6 = hop(ax, 5.7, yc - h / 2, 1.0, h, "Local update", [], MAU["hoc"]); the(ax, b6, "III-E")
    for a, b in ((b1, b2), (b2, b3), (b3, bb), (bb, b5), (b5, b6)):
        mui(ax, (a[1], yc), (b[0], yc))
    duong(ax, [s[1], 1.12], [s[5], s[5]]); duong(ax, [1.12, 1.12], [s[5], yc]); mui(ax, (1.12, yc), (b1[0], yc))

    sv = hop(ax, 5.3, 0.08, 1.4, 0.48, "Aggregation", ["at the server"], MAU["may_chu"]); the(ax, sv, "III-E")
    xs = b6[4]
    mui(ax, (xs, Y0), (xs, sv[3]), "site models", lech=(0.06, 0), ha="left")
    so(ax, xs - 0.1, (Y0 + sv[3]) / 2, 3)
    so(ax, sv[0] + 0.07, sv[3] - 0.07, 2)
    xr = bb[4]
    duong(ax, [sv[0], xr], [0.47, 0.47]); mui(ax, (xr, 0.47), (xr, Y0))
    ax.text(xr - 0.05, 0.72, "global model, next round", ha="right", va="center", fontsize=CO_CHU, style="italic")
    g = hop(ax, 3.05, 0.06, 1.05, 0.34, "Final global model", [], MAU["mang"]); the(ax, g, "III-G")
    mui(ax, (sv[0], 0.23), (g[1], 0.23))
    e1 = hop(ax, 1.2, 0.44, 1.6, 0.32, "Accuracy", [], MAU["danh_gia"]); the(ax, e1, "III-A")
    e2 = hop(ax, 1.2, 0.04, 1.6, 0.32, "Membership attack", [], MAU["rieng_tu"]); the(ax, e2, "III-F")
    duong(ax, [g[0], 2.92], [0.23, 0.23]); duong(ax, [2.92, 2.92], [e1[5], e2[5]])
    mui(ax, (2.92, e1[5]), (e1[1], e1[5])); mui(ax, (2.92, e2[5]), (e2[1], e2[5]))
    so(ax, e2[0] + 0.07, e2[3] - 0.07, 4)
    luu(fig, "hinh1_khung")


if __name__ == "__main__":
    hinh1(); hinh3(); hinh1_v3()
    print(f"Đã ghi vào {RA.resolve()}")
