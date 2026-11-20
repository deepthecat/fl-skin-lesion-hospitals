"""
kiem_do_nguon.py — chạy lại 2 con số trong paper (chỉ CPU, khoảng 1 phút, dùng bộ đệm ảnh đã có):
  (1) chỉ ảnh nốt ruồi lành (NV), 6 thống kê màu vùng giữa ảnh -> đoán 1 trong 4 nguồn. Paper ghi 0.80.
  (2) chỉ NV của hai nguồn Vienna lớn (molemax, modern), 6 thống kê màu của DẢI DA LÀNH trên và dưới ảnh
      (bỏ vùng tổn thương ở giữa và 4 góc) -> đoán 1 trong 2 nguồn. Paper ghi 0.99.
Cùng cách đánh giá như notebook 03: hồi quy logistic, 5 phần chia theo tổn thương, balanced accuracy.

    cd ~/ham_fl && source .venv/bin/activate && python kiem_do_nguon.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from du_lieu import RA, nap_dem

cs = pd.read_csv(RA / "chia_tap.csv")
ng = pd.read_csv(RA / "nguon.csv")[["image", "nguon"]]
vt = np.flatnonzero(cs.image.isin(ng.image))
tv = cs.iloc[vt].drop(columns=["nguon"], errors="ignore").merge(ng, on="image", how="left")   # nguồn lấy từ nguon.csv như notebook 03
assert tv.nguon.notna().all() and len(tv) == len(vt)
X = nap_dem(cs, 224, "p0")[vt]                      # đúng bộ đệm notebook 03 dùng


def thong_ke(v):
    v = v.astype(np.float32)
    return np.column_stack([v.mean((1, 2)), v.std((1, 2))])


def thu(F, d, nguon):
    y = d.nguon.map({s: i for i, s in enumerate(nguon)}).values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    yh = cross_val_predict(m, F, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), groups=d.lesion_id.values)
    return balanced_accuracy_score(y, yh)


NGUON = ["vidir_molemax", "vidir_modern", "rosendahl", "vienna_dias"]
nv = (tv.lop == "NV").values
F_giua = thong_ke(X[nv][:, 56:168, 56:168])
print(f"(1) NV, màu vùng giữa, 4 nguồn:          {thu(F_giua, tv[nv], NGUON):.3f}   (paper: 0.80, đoán bừa 0.25)")

V = ["vidir_molemax", "vidir_modern"]
m = nv & tv.nguon.isin(V).values
Xv = X[m]
for cao in (16, 24, 32):
    dai = np.concatenate([Xv[:, :cao, 32:192], Xv[:, -cao:, 32:192]], axis=1)   # dải trên + dưới, bỏ 4 góc
    print(f"(2) NV Vienna, dải da lành cao {cao:>2} px:     {thu(thong_ke(dai), tv[m], V):.3f}   (paper: 0.99, đoán bừa 0.50)")
print(f"    số ảnh: molemax {int((tv[m].nguon == V[0]).sum())}, modern {int((tv[m].nguon == V[1]).sum())}")
