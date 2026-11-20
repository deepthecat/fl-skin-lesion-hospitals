"""
tan_cong_thanh_vien.py — RQ2: mô hình rò rỉ thông tin về dữ liệu huấn luyện tới đâu?

    python tan_cong_thanh_vien.py

Kịch bản tấn công: kẻ tấn công có mô hình đã huấn luyện và một ảnh cụ thể. Hắn muốn biết ảnh đó CÓ
nằm trong tập huấn luyện không. Với dữ liệu y tế, chỉ riêng việc biết "bệnh nhân này có trong dữ liệu
của bệnh viện ung bướu" đã là rò rỉ.

Cách tấn công đơn giản nhất (Yeom và c.s., 2018): mô hình thường "thuộc bài" — loss trên ảnh đã học thấp
hơn loss trên ảnh chưa thấy. Nên đoán "là thành viên" khi loss thấp. Đây là tấn công YẾU NHẤT, nên con
số ra là CẬN DƯỚI của mức rò rỉ thật: tấn công mạnh hơn chỉ có thể tìm ra nhiều hơn.

  thành viên     = ảnh 'train'
  không thành viên = ảnh 'val' — CÙNG phân phối với train (cùng tập Training gốc), nhưng khác tổn thương
                     và mô hình chưa từng học. Không dùng tập test làm "không thành viên": test lệch phân
                     phối so với train, kẻ tấn công sẽ phân biệt được nhờ độ lệch đó chứ không phải nhờ
                     rò rỉ, và con số bị thổi phồng.

Ba chỉ số:
  AUC            — 0.5 = không phân biệt được gì (kín hoàn toàn), 1.0 = phân biệt hoàn hảo.
  lợi thế        — max(TPR - FPR). Đọc là: kẻ tấn công đoán đúng hơn tung đồng xu bao nhiêu.
  TPR@1%FPR      — chỉ số Carlini và c.s. (2022) khuyên dùng: khi chỉ được phép báo nhầm 1% số người
                   vô can, kẻ tấn công nhận ra được bao nhiêu phần trăm thành viên thật. Đây là con số
                   đáng sợ nhất, vì nó đo chuyện lộ CHẮC CHẮN một số bệnh nhân cụ thể.
  AUC theo lớp   — lớp hiếm vốn có loss cao hơn; tính AUC riêng trong từng lớp rồi lấy trung bình để
                   không nhầm "lớp khó" với "không phải thành viên".
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from du_lieu import LOP, RA


def loss_tung_anh(logit, y):
    z = logit - logit.max(1, keepdims=True)
    return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))


def chi_so(diem, la_tv):
    auc = roc_auc_score(la_tv, diem)
    fpr, tpr, _ = roc_curve(la_tv, diem)
    return auc, float(np.max(tpr - fpr)), float(np.interp(0.01, fpr, tpr))


def main():
    hang = []
    for f in sorted(glob.glob(str(RA / "*" / "*_logit.npz"))):
        z = np.load(f)
        phan, y, logit = z["phan"], z["nhan"], z["logit"].astype(np.float64)
        tv, ktv = phan == "train", phan == "val"
        chon = tv | ktv
        diem = -loss_tung_anh(logit[chon], y[chon])          # loss thấp -> điểm cao -> "là thành viên"
        la_tv = tv[chon].astype(int)
        auc, loi_the, tpr1 = chi_so(diem, la_tv)
        auc_lop = []
        for c in range(len(LOP)):
            m = y[chon] == c
            if m.any() and la_tv[m].min() != la_tv[m].max():  # lớp phải có mặt, và có cả hai phía
                auc_lop.append(roc_auc_score(la_tv[m], diem[m]))
        yh = logit.argmax(1)
        acc_tr, acc_v = (yh[tv] == y[tv]).mean(), (yh[ktv] == y[ktv]).mean()
        hang.append(dict(mo_hinh=Path(f).name.replace("_logit.npz", ""), loai=Path(f).parent.name,
                         AUC=auc, loi_the=loi_the, TPR_1FPR=tpr1, AUC_theo_lop=float(np.mean(auc_lop)),
                         acc_train=acc_tr, acc_val=acc_v, khoang_cach=acc_tr - acc_v))
    if not hang:
        raise SystemExit("Không thấy file *_logit.npz nào trong ketqua/")
    bang = pd.DataFrame(hang).sort_values(["loai", "mo_hinh"])
    bang.to_csv(RA / "tan_cong_thanh_vien.csv", index=False)
    pd.set_option("display.width", 200)
    print(bang.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nĐọc: AUC 0.5 = kín. 'khoang_cach' = acc train trừ acc val — mô hình càng học thuộc thì khoảng")
    print("này càng lớn và tấn công càng dễ. Đây là tấn công yếu nhất: rò rỉ thật chỉ có thể LỚN HƠN.")
    print(f"\nĐã ghi {RA / 'tan_cong_thanh_vien.csv'}")


if __name__ == "__main__":
    main()
