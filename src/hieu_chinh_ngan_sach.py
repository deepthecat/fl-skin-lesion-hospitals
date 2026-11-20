"""
hieu_chinh_ngan_sach.py — máy chủ cần bao nhiêu ảnh có nhãn để hiệu chỉnh ngưỡng còn tác dụng?

    python hieu_chinh_ngan_sach.py

Kết quả trước: hiệu chỉnh 7 độ lệch (cách C) lấy lại 92% khoảng mất mát ở dir0.3 — nhưng dùng TOÀN BỘ
1.432 ảnh validation có nhãn. Bệnh viện điều phối thực tế khó có nhiều vậy. Script này thu nhỏ tập của máy
chủ xuống N ảnh và đo phần lấy lại còn bao nhiêu.

Cách lấy N ảnh: mỗi lớp ít nhất 1 ảnh, phần còn lại chia theo tỉ lệ lớp — giống một bệnh viện điều phối
tự gom một tập nhỏ có đủ các loại bệnh. Mỗi N rút ngẫu nhiên 20 lần. Test chỉ dùng để báo cáo.

Chỉ chạy trên các mô hình FedAvg EfficientNet-B0 và mô hình tập trung EfficientNet-B0 (đủ để thấy xu hướng,
và nhanh). Vài phút trên CPU.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from du_lieu import LOP, RA
from hieu_chinh_nguong import chinh_do_lech

NGAN_SACH = (25, 50, 100, 200, 400, 800, None)      # None = toàn bộ validation
LAN = 20


def bacc(y, yh):
    return float(np.mean([(yh[y == c] == c).mean() for c in range(len(LOP)) if (y == c).any()]))


def rut(yv, n, rng):
    """Mỗi lớp ≥ 1 ảnh, còn lại theo tỉ lệ."""
    if n is None or n >= len(yv):
        return np.arange(len(yv))
    chon = []
    ti_le = np.bincount(yv, minlength=len(LOP)) / len(yv)
    for c in range(len(LOP)):
        o = np.flatnonzero(yv == c)
        k = min(len(o), max(1, int(round(n * ti_le[c]))))
        chon.append(rng.choice(o, k, replace=False))
    return np.concatenate(chon)


def nhom_cua(f):
    j = json.load(open(f.replace("_logit.npz", ".json")))
    c = j["cau_hinh"]
    return f"{c.get('thuat_toan', 'tập trung')} | {c.get('kich_ban', '—')}", c["seed"]


def main():
    tep = [f for f in sorted(glob.glob(str(RA / "*" / "*_logit.npz")))
           if "efficientnet_b0" in f and ("/tap_trung/" in f or "/lien_ket/fedavg_" in f)
           and "mot_may" not in f and "iid_anh" not in f and "_db" not in f and "_txl" not in f]
    if not tep:
        raise SystemExit("Không thấy logit nào phù hợp.")
    hang = []
    for f in tep:
        nhom, seed = nhom_cua(f)
        z = np.load(f)
        phan, y, lg = z["phan"], z["nhan"], z["logit"].astype(np.float64)
        v, t = phan == "val", phan == "test"
        lv, yv, lt, yt = lg[v], y[v], lg[t], y[t]
        goc = bacc(yt, lt.argmax(1))
        rng = np.random.default_rng(seed)
        for n in NGAN_SACH:
            lap = 1 if n is None else LAN
            for _ in range(lap):
                i = rut(yv, n, rng)
                _, lt_c = chinh_do_lech(lv[i], yv[i], lt)
                hang.append(dict(nhom=nhom, seed=seed, N=n or len(yv), lay_lai=bacc(yt, lt_c.argmax(1)) - goc))
        print(f"  xong {Path(f).name}", flush=True)

    ds = pd.DataFrame(hang)
    ds.to_csv(RA / "hieu_chinh_ngan_sach.csv", index=False)
    bang = ds.groupby(["nhom", "N"]).lay_lai.agg(["mean", "std"]).reset_index()
    bang["o"] = bang.apply(lambda r: f"{r['mean']:+.3f} ± {r['std']:.3f}", axis=1)
    thu_tu = {"tập trung | —": 0, "fedavg | iid": 1, "fedavg | dir1.0": 2, "fedavg | dir0.3": 3, "fedavg | dir0.1": 4}
    bang = bang.pivot(index="nhom", columns="N", values="o")
    bang = bang.loc[sorted(bang.index, key=lambda k: thu_tu.get(k, 9))]
    pd.set_option("display.width", 220)
    print("\nPHẦN TEST BACC LẤY LẠI THEO SỐ ẢNH CÓ NHÃN Ở MÁY CHỦ (trung bình ± độ lệch qua seed × lần rút)")
    print(bang.to_string())
    print(f"\nĐã ghi {RA / 'hieu_chinh_ngan_sach.csv'}")


if __name__ == "__main__":
    main()
