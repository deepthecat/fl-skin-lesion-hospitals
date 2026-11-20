"""
xem_du_lieu.py — buổi học 1: nhìn tận mắt dữ liệu trước khi đọc report (mục III-A, III-B).

    python xem_du_lieu.py

Không cần GPU, chạy dưới 1 phút. Vẽ 3 hình, lưu vào thư mục Windows Research_2026_Code/hinh_hoc
(mở bằng File Explorer) và vào ketqua/hinh_hoc:

  H1_bon_nguon_cung_benh.png  — cùng MỘT bệnh (NV, nốt ruồi lành) nhưng chụp ở 4 nguồn khác nhau.
                                 Nhìn màu nền, độ sáng, viền đen ở góc: khác nhau theo HÀNG = theo máy chụp.
  H2_bay_loai_benh.png        — 7 loại bệnh, mỗi loại 6 ảnh, kèm số ảnh. Thấy NV áp đảo thế nào.
  H3_cung_mot_ton_thuong.png  — mỗi hàng là NHIỀU ảnh của CÙNG một nốt trên da. Nếu chia ảnh ngẫu nhiên,
                                 ảnh này vào train, ảnh kia vào val -> mô hình "đã thấy đáp án".
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from du_lieu import LOP, RA, doc_anh, doc_chia_tap

TEN_VI = {"MEL": "melanoma (ung thư hắc tố)", "NV": "nốt ruồi lành", "BCC": "ung thư tế bào đáy",
          "AKIEC": "dày sừng quang hoá / tiền ung thư", "BKL": "dày sừng lành tính",
          "DF": "u xơ da", "VASC": "tổn thương mạch máu"}
NGUON = ["vidir_molemax", "vidir_modern", "rosendahl", "vienna_dias"]
WIN = Path("/mnt/c/Users/admin/Downloads/Research_2026_Code")
S = 160


def luu(fig, ten, noi):
    for d in noi:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / ten, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  đã vẽ", ten)


def luoi(hang, nhan_hang, tieu_de, so_cot=6):
    fig, ax = plt.subplots(len(hang), so_cot, figsize=(so_cot * 1.9, len(hang) * 2.05))
    ax = np.atleast_2d(ax)
    for r, (duong, nhan) in enumerate(zip(hang, nhan_hang)):
        for c in range(so_cot):
            a = ax[r, c]; a.axis("off")
            if c < len(duong):
                a.imshow(doc_anh(duong[c], S, "p1"))
        ax[r, 0].text(-0.08, 0.5, nhan, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle(tieu_de, fontsize=11, y=0.99)
    plt.subplots_adjust(wspace=0.03, hspace=0.06, top=1 - 0.35 / (len(hang) * 2.05))
    return fig


def main():
    rng = np.random.default_rng(0)
    ds = doc_chia_tap()
    ds = ds[ds.phan.isin(["train", "val"]) & ds.nguon.notna()]
    noi = [RA / "hinh_hoc"] + ([WIN / "hinh_hoc"] if WIN.exists() else [])

    # H1: cùng bệnh NV, 4 nguồn
    hang, nhan = [], []
    for n in NGUON:
        d = ds[(ds.nguon == n) & (ds.lop == "NV")]
        d1 = d.drop_duplicates("lesion_id")   # mỗi ô một nốt khác nhau
        hang.append(d1.duong_dan.sample(min(6, len(d1)), random_state=0).tolist())
        nhan.append(f"{n}\n{(ds.nguon == n).sum()} ảnh\n({(d.shape[0] / max((ds.nguon == n).sum(), 1)):.0%} là NV)")
    luu(luoi(hang, nhan, "Cùng bệnh NV, bốn nguồn chụp khác nhau — mỗi hàng một nguồn"), "H1_bon_nguon_cung_benh.png", noi)

    # H2: 7 loại bệnh
    hang, nhan = [], []
    for l in LOP:
        d = ds[ds.lop == l]
        d1 = d.drop_duplicates("lesion_id")   # mỗi ô một nốt khác nhau
        hang.append(d1.duong_dan.sample(min(6, len(d1)), random_state=0).tolist())
        nhan.append(f"{l}: {TEN_VI[l]}\n{len(d)} ảnh ({len(d) / len(ds):.0%})")
    luu(luoi(hang, nhan, "Bảy loại bệnh trong HAM10000 (train + val)"), "H2_bay_loai_benh.png", noi)

    # H3: cùng một tổn thương, nhiều ảnh
    dem = ds.groupby("lesion_id").size()
    nhieu = dem[dem >= 3].index.to_numpy()
    chon = rng.choice(nhieu, size=min(4, len(nhieu)), replace=False)
    hang = [ds[ds.lesion_id == t].duong_dan.tolist()[:6] for t in chon]
    nhan = [f"{t}\n{dem[t]} ảnh, {ds[ds.lesion_id == t].lop.iloc[0]}" for t in chon]
    so_ton_thuong, so_anh = ds.lesion_id.nunique(), len(ds)
    luu(luoi(hang, nhan, f"Mỗi hàng: các ảnh của CÙNG một nốt trên da "
                         f"({so_anh} ảnh nhưng chỉ {so_ton_thuong} nốt khác nhau)"), "H3_cung_mot_ton_thuong.png", noi)

    print("\nMở các hình ở:", *noi, sep="\n  ")


if __name__ == "__main__":
    main()
