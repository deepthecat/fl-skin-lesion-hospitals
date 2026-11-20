"""
kiem_trung_bcn.py — trước khi dùng ảnh BCN (ISIC 2019, trung tâm 0) làm dữ liệu CÔNG KHAI để học trước, kiểm không có
ảnh nào trùng với BẤT KỲ ảnh HAM10000 nào (train / val / test). Cùng cách hai tầng như kiem_trung_test.py:
  1. ứng viên: pHash cách nhau ≤ 10 bit;  2. xác nhận: tương quan chi tiết ≥ 0,80 (mức sàn của kiem_trung_test.py).
Ghi ketqua/bcn_loai_tru.csv (cột image) = ảnh BCN bị loại (đã xác nhận trùng). Ứng viên chưa xác nhận chỉ báo, không loại.
Lý do: dữ liệu "công khai" mà chứa bản sao ảnh của bệnh nhân HAM thì bảo đảm DP về sau không còn đúng nghĩa.
"""
import time

import numpy as np
import pandas as pd

from du_lieu import RA, doc_chia_tap
from fed_isic import ANH
from kiem_trung_test import dac_trung

NGUONG = 0.80
LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def main():
    t0 = time.time()
    ds = doc_chia_tap()
    cs = pd.read_csv(RA / "fed_isic_chia.csv")
    bcn = cs[(cs.center == 0) & (cs.phan == "train")].reset_index(drop=True)
    th = ANH["goc"]
    print(f"HAM10000: {len(ds)} ảnh · BCN train: {len(bcn)} ảnh ({th})", flush=True)
    dh = [dac_trung(p) for p in ds.duong_dan]
    Hh = np.array([x[0] for x in dh], np.uint64); Vh = np.stack([x[1] for x in dh])
    db = [dac_trung(th / f"{t}.jpg") for t in bcn.image]
    Hb = np.array([x[0] for x in db], np.uint64); Vb = np.stack([x[1] for x in db])
    print(f"Đặc trưng xong: {time.time() - t0:.0f}s", flush=True)
    hang = []
    for i0 in range(0, len(Hb), 256):
        x = (Hb[i0:i0 + 256, None] ^ Hh[None, :]).view(np.uint8).reshape(-1, len(Hh), 8)
        d = LUT[x].sum(-1)                                       # khoảng cách Hamming, (lô × số ảnh HAM)
        for a, b in zip(*np.nonzero(d <= 10)):
            hang.append(dict(image=bcn.image.iat[i0 + a], anh_ham=ds.image.iat[b], phan_ham=ds.phan.iat[b],
                             bit=int(d[a, b]), tuong_quan=float(Vb[i0 + a] @ Vh[b])))
    kq = pd.DataFrame(hang, columns=["image", "anh_ham", "phan_ham", "bit", "tuong_quan"])
    xn = kq[kq.tuong_quan >= NGUONG]
    xn[["image"]].drop_duplicates().to_csv(RA / "bcn_loai_tru.csv", index=False)
    kq.sort_values("tuong_quan", ascending=False).to_csv(RA / "bcn_trung_ung_vien.csv", index=False)
    print(f"Cặp ứng viên (≤ 10 bit): {len(kq)} · ảnh BCN ứng viên: {kq.image.nunique()}")
    print(f"XÁC NHẬN TRÙNG (tương quan ≥ {NGUONG}): {xn.image.nunique()} ảnh BCN → loại khỏi dữ liệu công khai")
    if len(kq):
        print("Tương quan cao nhất:", kq.tuong_quan.max().round(3))
    print(f"Đã ghi {RA / 'bcn_loai_tru.csv'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
