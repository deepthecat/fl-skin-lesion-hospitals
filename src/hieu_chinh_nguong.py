"""
hieu_chinh_nguong.py — kiểm giả thuyết: federated non-IID làm hỏng NGƯỠNG quyết định chứ không hỏng
khả năng NHÌN của mô hình.

    python hieu_chinh_nguong.py

Căn cứ: ở dir0.3, balanced accuracy rơi 0.685 -> 0.562 nhưng AUC gần như đứng yên (0.945 -> 0.938).
AUC đo khả năng xếp hạng, bacc đo quyết định cuối. Hai số tách nhau = mô hình vẫn phân biệt tốt, nhưng
quyết định bị lệch về lớp phổ biến. Nếu đúng, chỉnh lại ngưỡng ở máy chủ — KHÔNG huấn luyện lại — sẽ lấy
lại phần lớn khoảng đã mất.

Ba cách chỉnh, từ ít giả định tới nhiều giả định:
  A. logit - log(tần suất lớp)          — không tham số. Chỉ cần máy chủ biết phân bố lớp chung
                                          (đã giả định sẵn khi tính trọng số lớp). Menon và c.s., 2021.
  B. logit - τ·log(tần suất lớp)        — một tham số τ, chọn trên validation.
  C. logit + b_c, bảy độ lệch riêng     — chọn trên validation. Mạnh nhất, dễ khớp quá validation nhất.

B và C cần máy chủ giữ một tập validation có nhãn — một giả định phải ghi rõ trong bài.
Mọi tham số chỉ chọn trên VALIDATION. Test chỉ dùng để báo cáo.

Nếu giả thuyết đúng: khoảng lấy lại được ở các mô hình non-IID phải lớn hơn hẳn ở mô hình tập trung/IID.
Nếu sai: cả hai lấy lại được ngang nhau, và độ sụt ở non-IID là do mô hình nhìn kém thật.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from du_lieu import LOP, RA


def bacc(y, yh):
    return float(np.mean([(yh[y == c] == c).mean() for c in range(len(LOP)) if (y == c).any()]))


def chinh_tau(lv, yv, lt, logp):
    """B: chọn τ tốt nhất trên val."""
    luoi = np.linspace(-1, 3, 41)
    diem = [bacc(yv, (lv - t * logp).argmax(1)) for t in luoi]
    t = luoi[int(np.argmax(diem))]
    return t, lt - t * logp


def chinh_do_lech(lv, yv, lt, vong=3):
    """C: leo đồi từng lớp một, mỗi lớp thử 61 mức lệch, lặp 3 lượt."""
    b = np.zeros(len(LOP))
    hien = bacc(yv, lv.argmax(1))
    for _ in range(vong):
        for c in range(len(LOP)):
            tot_d, tot_s = 0.0, hien
            for d in np.linspace(-3, 3, 61):
                bb = b.copy(); bb[c] += d
                s = bacc(yv, (lv + bb).argmax(1))
                if s > tot_s + 1e-9 or (abs(s - tot_s) < 1e-9 and abs(d) < abs(tot_d)):
                    tot_d, tot_s = d, s
            b[c] += tot_d
            hien = tot_s
    return b, lt + b


def main():
    hang = []
    for f in sorted(glob.glob(str(RA / "*" / "*_logit.npz"))):
        z = np.load(f)
        phan, y, logit = z["phan"], z["nhan"], z["logit"].astype(np.float64)
        tr, v, t = phan == "train", phan == "val", phan == "test"
        pi = np.bincount(y[tr], minlength=len(LOP)) / tr.sum()
        logp = np.log(np.maximum(pi, 1e-8))
        lv, yv, lt, yt = logit[v], y[v], logit[t], y[t]

        goc = bacc(yt, lt.argmax(1))
        a = bacc(yt, (lt - logp).argmax(1))
        tau, lt_b = chinh_tau(lv, yv, lt, logp)
        b_ = bacc(yt, lt_b.argmax(1))
        _, lt_c = chinh_do_lech(lv, yv, lt)
        c_ = bacc(yt, lt_c.argmax(1))
        hang.append(dict(mo_hinh=Path(f).name.replace("_logit.npz", ""), loai=Path(f).parent.name,
                         test_goc=goc, A_khong_tham_so=a, B_tau=b_, tau=tau, C_7_do_lech=c_,
                         lay_lai_A=a - goc, lay_lai_C=c_ - goc,
                         ti_le_doan_NV=float((lt.argmax(1) == LOP.index("NV")).mean()),
                         ti_le_NV_that=float((yt == LOP.index("NV")).mean())))
    if not hang:
        raise SystemExit("Không thấy file *_logit.npz nào trong ketqua/")
    bang = pd.DataFrame(hang).sort_values(["loai", "mo_hinh"])
    bang.to_csv(RA / "hieu_chinh_nguong.csv", index=False)
    pd.set_option("display.width", 220)
    print(bang.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nĐọc: so cột lay_lai_* giữa các dòng tập trung / iid với các dòng dir0.3 / dir0.1.")
    print("ti_le_doan_NV >> ti_le_NV_that = mô hình đang dồn quyết định về lớp phổ biến.")
    print(f"\nĐã ghi {RA / 'hieu_chinh_nguong.csv'}")


if __name__ == "__main__":
    main()
