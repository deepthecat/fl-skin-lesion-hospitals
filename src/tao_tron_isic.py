"""
tao_tron_isic.py — kịch bản đối chứng TRỘN MÁY cho Fed-ISIC2019 (giống tao_kich_ban_tron.py của HAM10000).

    python tao_tron_isic.py

Mỗi "bệnh viện" giữ NGUYÊN số ảnh của từng lớp bệnh như phép chia thật của FLamby, nhưng ảnh của cùng một lớp được
trộn ngẫu nhiên giữa 6 trung tâm → kích thước và tỉ lệ bệnh không đổi, chỉ có MÁY CHỤP bị trộn đều.
Chỉ đụng tới ảnh train; tập test giữ nguyên theo trung tâm thật. Thêm cột `may_tron` vào ketqua/fed_isic_chia.csv.
"""
import numpy as np
import pandas as pd

from du_lieu import RA

f = RA / "fed_isic_chia.csv"
cs = pd.read_csv(f)
tr = cs.phan == "train"
rng = np.random.default_rng(0)
moi = pd.Series(np.nan, index=cs.index)
for lop, nhom in cs[tr].groupby("target"):
    tram = nhom.center.values.copy()            # số ảnh lớp này ở mỗi trung tâm được giữ nguyên
    moi[nhom.index] = rng.permutation(tram)     # nhưng ảnh nào về trung tâm nào thì xáo ngẫu nhiên
cs["may_tron"] = moi
cs.to_csv(f, index=False)
t = cs[tr]
print("Số ảnh mỗi client (thật vs trộn) — phải bằng nhau:")
print(pd.DataFrame({"thật": t.center.value_counts().sort_index(), "trộn": t.may_tron.astype(int).value_counts().sort_index()}))
print("\nTỉ lệ trung tâm gốc trong mỗi client trộn (hàng = client trộn):")
print(pd.crosstab(t.may_tron.astype(int), t.center, normalize="index").round(2))
