"""
tao_kich_ban_tron.py — kịch bản đối chứng `tron_nguon` (thử thách 1: tách máy chụp khỏi kích thước + nhãn).

    python tao_kich_ban_tron.py                 # cột may_tron_nguon  (lần trộn seed 0)
    python tao_kich_ban_tron.py --seed-tron 1   # cột may_tron_nguon1 (lần trộn khác, để đo độ dao động do cách trộn)

Ý tưởng: giữ gần như NGUYÊN số ảnh của từng lớp ở từng bệnh viện như kịch bản `nguon`
(nên kích thước và tỉ lệ bệnh của mỗi bệnh viện gần như y hệt), nhưng tổn thương được rút
ngẫu nhiên từ cả 4 nguồn (cả tổn thương đi cùng nhau, không xé lẻ). Mỗi "bệnh viện" giờ có ảnh của mọi máy chụp -> khác biệt máy chụp
giữa các bệnh viện bị xoá, còn lệch kích thước và lệch nhãn được giữ.

So FedAvg trên `nguon` với `tron_nguon`:
  - nếu khoảng cách tụt mạnh  -> phần lớn thiệt hại đến từ máy chụp;
  - nếu gần như không đổi       -> thiệt hại đến từ kích thước / nhãn.

Chỉ đụng tới các dòng train; thêm cột `may_tron_nguon` vào ketqua/chia_tap.csv
(bản cũ lưu ở chia_tap_truoc_tron.csv). Chạy lại nhiều lần cho cùng kết quả (seed cố định).
"""
import argparse
import shutil

import numpy as np
import pandas as pd

from du_lieu import RA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-tron", type=int, default=0)
    SEED = ap.parse_args().seed_tron
    COT = "may_tron_nguon" if SEED == 0 else f"may_tron_nguon{SEED}"
    f = RA / "chia_tap.csv"
    cs = pd.read_csv(f)
    nhan = "nhan" if "nhan" in cs.columns else "lop"
    for c in ("phan", "lesion_id", "may_nguon", nhan):
        if c not in cs.columns:
            raise SystemExit(f"Thiếu cột {c} trong chia_tap.csv — chạy notebook 02 và 06 trước.")
    if not (RA / "chia_tap_truoc_tron.csv").exists():
        shutil.copy(f, RA / "chia_tap_truoc_tron.csv")

    tr = cs[cs.phan == "train"]
    tt = tr.groupby("lesion_id").agg(lop=(nhan, "first"), may=("may_nguon", "first"),
                                     n=("image", "size")).reset_index()
    if tt.may.isna().any():
        raise SystemExit("Có tổn thương train chưa có may_nguon.")
    tt["may"] = tt.may.astype(int)

    rng = np.random.default_rng(SEED)
    moi = np.empty(len(tt), int)
    for c in sorted(tt.lop.unique()):
        idx = np.flatnonzero(tt.lop.values == c)
        # mục tiêu: số ẢNH lớp c ở mỗi bệnh viện thật (khớp ảnh chứ không khớp tổn thương,
        # vì số ảnh mỗi tổn thương khác nhau giữa các nguồn)
        muc_tieu = np.bincount(tt.may.values[idx], weights=tt.n.values[idx], minlength=4).astype(float)
        da_co = np.zeros(4)
        for i in rng.permutation(idx):                          # rút từ cả 4 nguồn, thứ tự ngẫu nhiên
            thieu = (muc_tieu - da_co) / np.maximum(muc_tieu, 1)   # bệnh viện thiếu nhiều nhất (tương đối) nhận trước
            thieu[muc_tieu == 0] = -np.inf
            k = int(np.argmax(thieu))
            moi[i] = k
            da_co[k] += tt.n.values[i]
    tt["moi"] = moi

    cs = cs.drop(columns=[c for c in (COT,) if c in cs.columns])
    cs[COT] = cs.lesion_id.map(dict(zip(tt.lesion_id, tt.moi))).where(cs.phan == "train")
    cs.to_csv(f, index=False)

    tr = cs[cs.phan == "train"]
    print(f"Số ảnh mỗi bệnh viện (nguon thật vs {COT[4:]}):")
    print(pd.DataFrame({"nguon": tr.may_nguon.value_counts().sort_index(),
                        COT[4:]: tr[COT].value_counts().sort_index()}).astype(int).to_string())
    if "nguon" in cs.columns:
        print("\nTỉ lệ nguồn gốc trong mỗi bệnh viện tron_nguon. Trong CÙNG một lớp bệnh các hàng giống nhau;"
              "\nhàng có nhiều NV sẽ có nhiều molemax hơn vì molemax vốn toàn NV — phần đó gắn với nhãn, không tách được:")
        print(pd.crosstab(tr[COT].astype(int), tr.nguon, normalize="index").round(2).to_string())
    print(f"\nĐã ghi cột {COT} vào {f}")


if __name__ == "__main__":
    main()
