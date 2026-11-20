"""
phan_tich_dp.py — H1: khi thêm DP, AI trả giá? Và DP có thu hẹp rủi ro bị lộ không?

    python phan_tich_dp.py              # P1
    python phan_tich_dp.py --txl p2     # P2

Đọc logit của dp_tuyen_tinh.py (ketqua/dp_tuyen_tinh/dptt_*_logit.npz). Không cần GPU, vài giây.
Mọi con số là trung bình 3 seed. "mất" = (không DP) − (có DP): càng lớn càng trả giá nặng.

  1. val balanced accuracy theo mức riêng tư, cho 3 kịch bản (tập trung / 4 nguồn thật / trộn máy).
  2. 4 nguồn thật: val bAcc theo từng bệnh viện — bệnh viện nào mất nhiều nhất khi thêm DP.
  3. 4 nguồn thật: recall từng loại bệnh trên val — bệnh nào mất nhiều nhất.
  4. AUC tấn công membership (ngưỡng loss) tổng và theo bệnh — DP có làm kín hơn không.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from du_lieu import LOP, RA
from phan_tich_theo_site import NGAN, NGUON
from tan_cong_thanh_vien import loss_tung_anh

warnings.filterwarnings("ignore")
MUC = ("inf", "clip", "8", "3", "1", "0.5")
KB = {"mot_may": "tập trung", "nguon": "4 nguồn thật", "tron_nguon": "trộn máy"}


def auc(la, d):
    return roc_auc_score(la, d) if 0 < la.sum() < len(la) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txl", default="p1")
    ap.add_argument("--kt", type=int, default=224)
    arg = ap.parse_args()
    cs = pd.read_csv(RA / "chia_tap.csv")
    nguon_cua = dict(zip(cs.image, cs.nguon))

    hang = []
    for kb in KB:
        for m in MUC:
            for s in (0, 1, 2):
                f = RA / "dp_tuyen_tinh" / f"dptt_{kb}_eps{m}_efficientnet_b0_txl{arg.txl}_kt{arg.kt}_s{s}_logit.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                phan, y, lg = d["phan"], d["nhan"], d["logit"].astype(np.float64)
                yh = lg.argmax(1)
                ng = np.array([nguon_cua.get(i, "") for i in d["image"]])
                v = phan == "val"
                r = dict(kb=kb, muc=m, seed=s, val=balanced_accuracy_score(y[v], yh[v]))
                for n in NGUON:
                    mm = v & (ng == n)
                    r[NGAN[n]] = balanced_accuracy_score(y[mm], yh[mm]) if mm.any() else np.nan
                for c, l in enumerate(LOP):
                    mm = v & (y == c)
                    r[f"rc_{l}"] = float(np.mean(yh[mm] == c)) if mm.any() else np.nan
                ch = (phan == "train") | v
                diem = -loss_tung_anh(lg[ch], y[ch])
                la = (phan[ch] == "train").astype(int)
                r["mia"] = auc(la, diem)
                for c, l in enumerate(LOP):
                    mm = y[ch] == c
                    r[f"mia_{l}"] = auc(la[mm], diem[mm])
                hang.append(r)
    if not hang:
        raise SystemExit(f"Không thấy file dptt_*_txl{arg.txl}_* nào trong ketqua/dp_tuyen_tinh.")
    B = pd.DataFrame(hang)
    B.to_csv(RA / f"phan_tich_dp_{arg.txl}.csv", index=False)
    G = B.groupby(["kb", "muc"]).mean(numeric_only=True)
    n = B.groupby(["kb", "muc"]).size()

    def bang(kb, cot, tieu_de, rong=9, hieu=True):
        if kb not in G.index.get_level_values(0):
            return
        print(tieu_de)
        print(f"{'mức ε':<8}{'n':>2}" + "".join(f"{c.replace('rc_', '').replace('mia_', ''):>{rong}}" for c in cot))
        goc = G.loc[(kb, "inf")] if (kb, "inf") in G.index else None
        for m in MUC:
            if (kb, m) not in G.index:
                continue
            g = G.loc[(kb, m)]
            print(f"{m:<8}{n[(kb, m)]:>2}" + "".join(f"{g[c]:>{rong}.3f}" for c in cot))
        if hieu and goc is not None:
            for m in ("8", "3", "1"):
                if (kb, m) in G.index:
                    g = G.loc[(kb, m)]
                    print(f"{'mất ε=' + m:<10}" + "".join(f"{goc[c] - g[c]:>{rong}.3f}" for c in cot))
        print()

    print(f"=== DP-SGD tầng cuối, đặc trưng ImageNet đóng băng, tiền xử lý {arg.txl.upper()} ===\n")
    print("1. VAL BALANCED ACCURACY")
    print(f"{'mức ε':<8}" + "".join(f"{v:>16}" for v in KB.values()))
    for m in MUC:
        print(f"{m:<8}" + "".join(f"{G.loc[(k, m), 'val']:>16.3f}" if (k, m) in G.index else f"{'—':>16}" for k in KB))
    print()
    bang("nguon", list(NGAN.values()), "2. 4 NGUỒN THẬT — VAL BACC THEO BỆNH VIỆN (val: 559 / 466 / 333 / 74 ảnh)", 11)
    bang("nguon", [f"rc_{l}" for l in LOP], "3. 4 NGUỒN THẬT — RECALL TỪNG BỆNH TRÊN VAL")
    for kb in ("mot_may", "nguon"):
        bang(kb, ["mia"] + [f"mia_{l}" for l in LOP],
             f"4. AUC TẤN CÔNG ({KB[kb]}) — 0.5 = kín; hàng 'mất' dương = DP làm kín hơn", 8)
    print(f"Đã ghi {RA / f'phan_tich_dp_{arg.txl}.csv'}")


if __name__ == "__main__":
    main()
