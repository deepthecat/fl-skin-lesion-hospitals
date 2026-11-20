"""
rmia.py — tấn công suy luận thành viên MẠNH HƠN (RMIA, bản offline) so với tấn công dựa vào loss. Không cần GPU.

    python rmia.py

Vì sao cần: tan_cong_thanh_vien.py dùng loss thô. Loss cao có thể vì ảnh KHÓ chứ không phải vì mô hình chưa thấy nó,
nên tấn công đó là cận dưới yếu. RMIA (Zarifzadeh và c.s., ICML 2024) so độ tự tin của mô hình nạn nhân trên ảnh x
với độ tự tin của các mô hình THAM CHIẾU chưa từng học x:
    tỉ số(x) = p_nạn_nhân(nhãn đúng | x) / p_tham_chiếu(x)
    p_tham_chiếu(x) = ½·[(1+a)·p_OUT(x) + (1−a)]   (bản offline: chỉ có mô hình tham chiếu KHÔNG học x)
Tỉ số càng lớn -> mô hình nạn nhân tự tin bất thường với x -> nhiều khả năng x là thành viên.
RMIA gốc còn so tỉ số của x với tỉ số của các ảnh quần thể z; phép so đó không đổi THỨ TỰ xếp hạng giữa các ảnh,
nên AUC và TPR@FPR (hai chỉ số mình báo) tính thẳng từ tỉ số là như nhau.
Hệ số a được CHỈNH TRÊN MÔ HÌNH THAM CHIẾU (coi một mô hình tham chiếu là nạn nhân giả), không nhìn mô hình nạn nhân thật.

Thành viên = ảnh 'train', không thành viên = ảnh 'val' — giống hệt tan_cong_thanh_vien.py để số so được với nhau.
Khoảng tin cậy 95%: bootstrap theo TỔN THƯƠNG, 200 lần.
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from du_lieu import LOP, RA, doc_chia_tap

warnings.filterwarnings("ignore")
MO_HINH = ("convnext_atto", "efficientnet_b0")
NAN_NHAN = {  # nhãn -> mẫu tên file logit của mô hình nạn nhân (3 seed)
    "FedAvg nguồn, 30 vòng": "lien_ket/fedavg_nguon_{bb}_txlp1_kt224_s{s}_logit.npz",
    "FedAvg nguồn, 50 vòng": "lien_ket/fedavg_nguon_{bb}_txlp1_v50_kt224_s{s}_logit.npz",
    "Tập trung": "tap_trung/{bb}_kt224_nghich_txlp1_s{s}_logit.npz",
}


def p_dung(logit, y):
    z = logit.astype(np.float64); z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p[np.arange(len(y)), y]


def chi_so(diem, la_tv):
    fpr, tpr, _ = roc_curve(la_tv, diem)
    return roc_auc_score(la_tv, diem), float(np.interp(0.01, fpr, tpr))


def auc_theo_lop(diem, la_tv, y):
    a = [roc_auc_score(la_tv[y == c], diem[y == c]) for c in range(len(LOP))
         if (y == c).any() and la_tv[y == c].min() != la_tv[y == c].max()]
    return float(np.mean(a))


def diem_rmia(p_nn, P_ref, trong_ref, a):
    """P_ref [K, N]: p nhãn đúng của K mô hình tham chiếu; trong_ref [K, N]: mô hình k đã học ảnh i chưa."""
    ngoai = ~trong_ref
    p_out = (P_ref * ngoai).sum(0) / np.maximum(ngoai.sum(0), 1)
    return p_nn / (0.5 * ((1 + a) * p_out + (1 - a)) + 1e-12)


def main():
    ds = doc_chia_tap()
    y_all, phan = ds.nhan.values, ds.phan.values
    chon = np.isin(phan, ["train", "val"])
    la_tv = (phan[chon] == "train").astype(int)
    y = y_all[chon]
    les = ds.lesion_id.values[chon]
    nhom = pd.Series(np.arange(len(les))).groupby(les).apply(np.array).values   # chỉ số theo từng tổn thương
    rng = np.random.default_rng(0)
    mau_bs = [np.concatenate(nhom[rng.integers(0, len(nhom), len(nhom))]) for _ in range(200)]

    def ci(diem):
        v = []
        for ii in mau_bs:
            if la_tv[ii].min() != la_tv[ii].max():
                v.append(roc_auc_score(la_tv[ii], diem[ii]))
        return np.percentile(v, [2.5, 97.5])

    hang = []
    for bb in MO_HINH:
        fs = sorted(glob.glob(str(RA / "tham_chieu" / f"tc_{bb}_txlp1_kt224_c*n*.npz")))
        if len(fs) < 2:
            print(f"[{bb}] chưa đủ mô hình tham chiếu ({len(fs)}) — bỏ qua."); continue
        R = [np.load(f) for f in fs]
        for r in R:
            assert (r["image"] == ds.image.values).all(), "thứ tự ảnh của mô hình tham chiếu lệch chia_tap.csv"
        P = np.stack([p_dung(r["logit"], y_all) for r in R])[:, chon]
        T = np.stack([r["trong"] for r in R])[:, chon]
        # kiểm tra: mô hình tham chiếu có 'nhớ' ảnh đã học không (p trên ảnh đã học > p trên ảnh chưa học)?
        trong_tb = (P * T).sum() / T.sum()
        ngoai_tr = (P * (~T) * (la_tv == 1)).sum() / ((~T) & (la_tv == 1)).sum()
        print(f"[{bb}] {len(R)} mô hình tham chiếu | p(nhãn đúng) TB trên ảnh train đã học {trong_tb:.3f} "
              f"vs ảnh train chưa học {ngoai_tr:.3f} (chênh càng lớn = càng học thuộc)")
        # chỉnh a trên nạn nhân giả = mô hình tham chiếu đầu tiên; thành viên giả = ảnh nó đã học, không TV = val
        gia = T[0] | (la_tv == 0)
        ltv_gia = T[0][gia].astype(int)
        luoi = np.round(np.arange(0, 1.01, 0.1), 1)
        auc_a = [roc_auc_score(ltv_gia, diem_rmia(P[0], P[1:], T[1:], a)[gia]) for a in luoi]
        a_tot = float(luoi[int(np.argmax(auc_a))])
        print(f"[{bb}] chỉnh a trên nạn nhân giả: a = {a_tot} (AUC giả {max(auc_a):.3f})")
        for nhan, mau in NAN_NHAN.items():
            for s in range(3):
                f = RA / mau.format(bb=bb, s=s)
                if not f.exists():
                    continue
                z = np.load(f)
                assert (z["image"] == ds.image.values).all(), f"thứ tự ảnh lệch: {f.name}"
                p_nn = p_dung(z["logit"], y_all)[chon]
                d_loss = np.log(p_nn + 1e-12)                       # = −loss: loss thấp -> điểm cao
                d_rmia = diem_rmia(p_nn, P, T, a_tot)
                (al, tl), (ar, tr) = chi_so(d_loss, la_tv), chi_so(d_rmia, la_tv)
                cl, cr = ci(d_loss), ci(d_rmia)
                hang.append(dict(mo_hinh=bb, nan_nhan=nhan, seed=s,
                                 AUC_loss=al, AUC_loss_ci=f"[{cl[0]:.3f};{cl[1]:.3f}]", TPR1_loss=tl,
                                 AUC_rmia=ar, AUC_rmia_ci=f"[{cr[0]:.3f};{cr[1]:.3f}]", TPR1_rmia=tr,
                                 AUC_lop_loss=auc_theo_lop(d_loss, la_tv, y), AUC_lop_rmia=auc_theo_lop(d_rmia, la_tv, y)))
    if not hang:
        raise SystemExit("Không có mô hình nạn nhân nào đủ dữ liệu.")
    b = pd.DataFrame(hang)
    b.to_csv(RA / "rmia.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nTheo từng seed:")
    print(b.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTrung bình 3 seed:")
    tb = b.groupby(["mo_hinh", "nan_nhan"])[["AUC_loss", "AUC_rmia", "TPR1_loss", "TPR1_rmia", "AUC_lop_loss", "AUC_lop_rmia"]].mean()
    print(tb.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nĐọc: AUC 0,5 = kín. TPR1 = phần trăm thành viên bị nhận ra CHẮC khi chỉ được báo nhầm 1% người vô can.")
    print("RMIA cao hơn loss = mức rò rỉ thật lớn hơn con số cũ. Vẫn là cận dưới (tham chiếu học tập trung, nạn nhân học liên kết).")
    print(f"Đã ghi {RA / 'rmia.csv'}")


if __name__ == "__main__":
    main()
