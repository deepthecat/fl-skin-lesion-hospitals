"""
kiem_trung_test.py — xác minh ảnh validation/test chính thức có trùng ảnh train không, BẰNG SỐ.

    python kiem_trung_test.py

Notebook 01 dùng pHash với ngưỡng rất chặt (≤4 bit) và tìm ra 31 cặp nghi trùng. Hash một mình không
đủ để kết luận: ảnh soi da có chung viền đen tròn và nền da, hai ảnh khác nhau có thể ra hash gần nhau.

Cách xác minh — hai tầng, giống cách đã dùng ở đề tài trước:
  1. Ứng viên: pHash cách nhau ≤ 10 bit (nới rộng hơn notebook 01 để không sót).
  2. Xác nhận: TƯƠNG QUAN CHI TIẾT. Thu ảnh về 48×48 xám, trừ đi bản làm mờ 7×7 (bỏ màu nền, bỏ viền,
     chỉ giữ cấu trúc nhỏ như mạng sắc tố, mạch máu), rồi tính tương quan. Hai ảnh khác nhau gần như
     không có cấu trúc nhỏ trùng khớp, nên tương quan thấp.

Ngưỡng tương quan KHÔNG vay từ đề tài cũ: script đo lại trên chính HAM10000 bằng hai mốc
  - cặp ngẫu nhiên không liên quan  -> phải thấp
  - cặp ảnh cùng tổn thương (khác lần chụp) -> phải cao hơn hẳn
rồi đặt ngưỡng ở trên phân vị 99,9% của cặp ngẫu nhiên.

Ghi kết quả: cột `trung_train` (0/1) vào ketqua/chia_tap.csv, và ketqua/trung_test_xac_minh.csv.
Script huấn luyện sẽ tự báo thêm điểm `test_sach` (bỏ các ảnh trùng) bên cạnh điểm test đầy đủ.
"""
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from scipy.ndimage import uniform_filter

from du_lieu import RA, doc_chia_tap


def dac_trung(p):
    with Image.open(p) as im:
        im.draft("L", (256, 256))
        g = im.convert("L")
        a = np.asarray(g.resize((32, 32), Image.LANCZOS), np.float64)
        b = np.asarray(g.resize((48, 48), Image.LANCZOS), np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()
    h = np.packbits(d > np.median(d[1:])).view(np.uint64)[0]
    v = (b - uniform_filter(b, 7, mode="nearest")).flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return h, (v / n if n > 0 else v).astype(np.float32)


def hamming(h, H):
    return np.unpackbits((H ^ h).view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def main():
    ds = doc_chia_tap()
    t0 = time.time()
    dt = [dac_trung(p) for p in ds.duong_dan]
    H = np.array([x[0] for x in dt], dtype=np.uint64)
    V = np.stack([x[1] for x in dt])
    print(f"Đặc trưng cho {len(ds)} ảnh: {time.time() - t0:.0f}s")

    la_train = ds.phan.isin(["train", "val"]).values        # toàn bộ tập Training gốc
    i_tr = np.flatnonzero(la_train)
    i_ngoai = np.flatnonzero(~la_train)                      # val chính thức + test

    # ---- mốc 1: cặp ngẫu nhiên trong train (gần như chắc chắn không liên quan)
    rng = np.random.default_rng(0)
    a, b = rng.choice(i_tr, 5000), rng.choice(i_tr, 5000)
    khac = ds.lesion_id.values[a] != ds.lesion_id.values[b]
    r_ngau = np.einsum("ij,ij->i", V[a[khac]], V[b[khac]])
    # ---- mốc 2: cặp ảnh CÙNG tổn thương (chụp khác lần -> không y hệt, nhưng liên quan)
    cap = []
    for _, g in ds.iloc[i_tr].groupby("lesion_id").groups.items():
        g = list(g)
        cap += [(g[x], g[y]) for x in range(len(g)) for y in range(x + 1, len(g))]
    cap = np.array(cap)
    r_cung = np.einsum("ij,ij->i", V[cap[:, 0]], V[cap[:, 1]]) if len(cap) else np.array([np.nan])

    q = lambda x: " / ".join(f"{np.nanpercentile(x, p):.2f}" for p in (50, 90, 99, 99.9))
    print(f"\nMỐC ĐO TRÊN CHÍNH HAM10000 (tương quan chi tiết; phân vị 50 / 90 / 99 / 99,9)")
    print(f"  cặp ngẫu nhiên không liên quan ({len(r_ngau)} cặp):  {q(r_ngau)}")
    print(f"  cặp cùng tổn thương, khác lần chụp ({len(r_cung)} cặp): {q(r_cung)}")
    nguong = max(0.8, float(np.percentile(r_ngau, 99.9)) + 0.05)
    print(f"  -> ngưỡng xác nhận trùng: {nguong:.2f}")

    # ---- xét từng ảnh val chính thức / test
    hang = []
    for i in i_ngoai:
        d = hamming(H[i], H[i_tr])
        ung = i_tr[d <= 10]
        if not len(ung):
            continue
        r = V[ung] @ V[i]
        j = int(np.argmax(r))
        k = ung[j]
        hang.append(dict(phan=ds.phan.iat[i], anh=ds.image.iat[i], anh_train=ds.image.iat[k],
                         bit=int(hamming(H[i], H[k:k + 1])[0]), tuong_quan=float(r[j]),
                         lop=ds.lop.iat[i], lop_train=ds.lop.iat[k],
                         cung_lop=bool(ds.lop.iat[i] == ds.lop.iat[k])))
    kq = pd.DataFrame(hang)
    if len(kq):
        kq["xac_nhan"] = kq.tuong_quan >= nguong
        kq = kq.sort_values("tuong_quan", ascending=False)
    kq.to_csv(RA / "trung_test_xac_minh.csv", index=False)

    print(f"\nỨNG VIÊN (pHash ≤ 10 bit): {len(kq)}")
    if len(kq):
        print(f"XÁC NHẬN TRÙNG (tương quan ≥ {nguong:.2f}): {int(kq.xac_nhan.sum())}")
        print(kq.groupby(["phan", "xac_nhan"]).size().rename("so_anh").to_string())
        xn = kq[kq.xac_nhan]
        if len(xn):
            print(f"\nTrong các cặp xác nhận: cùng chẩn đoán {int(xn.cung_lop.sum())} / {len(xn)}")
            print(xn.lop.value_counts().rename("lớp của ảnh trùng").to_string())
        print("\n15 cặp tương quan cao nhất:")
        print(kq.head(15)[["phan", "anh", "anh_train", "bit", "tuong_quan", "lop", "lop_train"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    trung = set(kq.loc[kq.xac_nhan, "anh"]) if len(kq) else set()
    ds["trung_train"] = ds.image.isin(trung).astype(int)
    ds.to_csv(RA / "chia_tap.csv", index=False)
    print(f"\nĐã ghi cột trung_train vào chia_tap.csv ({len(trung)} ảnh đánh dấu)")


if __name__ == "__main__":
    main()
