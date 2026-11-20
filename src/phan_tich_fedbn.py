"""
phan_tich_fedbn.py — đọc kết quả đêm 9: FedBN có trúng G8b không? Không cần GPU, vài giây.

    python phan_tich_fedbn.py

G8b: khác biệt máy chụp hại FL qua thống kê BatchNorm. Nếu đúng:
  (1) FedBN > FedAvg rõ rệt trên NGUỒN THẬT, nhưng gần như bằng FedAvg trên TRỘN MÁY (ở đó không còn khác máy);
  (2) thống kê BN giữa các máy trạm xa nhau hơn ở nguồn thật so với trộn máy (bn_kc);
  (3) rosendahl (máy khác nhất) được lợi nhiều nhất.
Val FedBN nguồn thật có 2 cách chấm: "theo nguồn" (ảnh của bệnh viện k chấm bằng mô hình k — cách dùng thật)
và "ensemble" (trung bình K mô hình — dùng khi không biết nguồn, như test). Trộn máy chỉ có ensemble.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from du_lieu import RA
from phan_tich_theo_site import NGAN, NGUON

SEED = (0, 1, 2)
CH = {
    "FedAvg nguồn P1": "fedavg_nguon_efficientnet_b0_txlp1_kt224_s{s}",
    "FedBN nguồn P1": "fedbn_nguon_efficientnet_b0_txlp1_gh_kt224_s{s}",
    "FedAvg trộn P1": "fedavg_tron_nguon_efficientnet_b0_txlp1_kt224_s{s}",
    "FedBN trộn P1": "fedbn_tron_nguon_efficientnet_b0_txlp1_gh_kt224_s{s}",
    "FedAvg nguồn P2": "fedavg_nguon_efficientnet_b0_txlp2_kt224_s{s}",
    "FedBN nguồn P2": "fedbn_nguon_efficientnet_b0_txlp2_gh_kt224_s{s}",
}
TAP_TRUNG = {"P1": 0.778, "P2": 0.781}


def ms(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if not len(x):
        return "—"
    return f"{x.mean():.3f}±{x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def main():
    cs = pd.read_csv(RA / "chia_tap.csv")
    nguon_cua = dict(zip(cs.image, cs.nguon))
    hang = []
    for ten, mau in CH.items():
        for s in SEED:
            f = RA / "lien_ket" / f"{mau.format(s=s)}.json"
            if not f.exists():
                continue
            d = json.load(open(f)); k = d["ket_qua"]
            r = dict(ch=ten, seed=s, val=k["val"]["bacc"], test=k["test"]["bacc"],
                     val_ens=k.get("val_ens", {}).get("bacc"), test_bn_tb=k.get("test_bn_tb", {}).get("bacc"),
                     bn_kc=np.mean([h["bn_kc_tb"] for h in d["lich_su"][-10:]]) if "bn_kc_tb" in d["lich_su"][-1] else None,
                     vong_tot=int(np.argmax([h["val_bacc"] for h in d["lich_su"]])) + 1)
            z = np.load(RA / "lien_ket" / f"{mau.format(s=s)}_logit.npz")
            ng = np.array([nguon_cua.get(i, "") for i in z["image"]]); v = z["phan"] == "val"
            yh = z["logit"].argmax(1)
            for n in NGUON:
                m = v & (ng == n)
                r[NGAN[n]] = balanced_accuracy_score(z["nhan"][m], yh[m])
            hang.append(r)
    if not hang:
        raise SystemExit("Chưa có kết quả nào.")
    B = pd.DataFrame(hang)

    print("1. ĐỘ CHÍNH XÁC (balanced accuracy, trung bình ± sd qua seed)")
    print("   val FedBN nguồn = chấm theo nguồn; 'val ens' = ensemble; 'test BN tb' = test với BN trung bình kiểu FedAvg\n")
    print(f"{'cấu hình':<17}{'n':>2}{'val':>13}{'khoảng':>8}{'val ens':>13}{'test':>13}{'test BN tb':>13}{'vòng tốt':>9}")
    for ten in CH:
        g = B[B.ch == ten]
        if not len(g):
            continue
        p = ten[-2:]
        print(f"{ten:<17}{len(g):>2}{ms(g.val):>13}{TAP_TRUNG[p] - g.val.mean():>8.3f}{ms(g.val_ens):>13}"
              f"{ms(g.test):>13}{ms(g.test_bn_tb):>13}{g.vong_tot.mean():>9.1f}")

    print("\n2. G8b(1) — FedBN − FedAvg (val). Dự đoán: nguồn thật dương rõ, trộn máy ≈ 0")
    for a, b in (("FedBN nguồn P1", "FedAvg nguồn P1"), ("FedBN trộn P1", "FedAvg trộn P1"), ("FedBN nguồn P2", "FedAvg nguồn P2")):
        ga, gb = B[B.ch == a], B[B.ch == b]
        if len(ga) and len(gb):
            hieu = [ga[ga.seed == s].val.iloc[0] - gb[gb.seed == s].val.iloc[0] for s in SEED
                    if (ga.seed == s).any() and (gb.seed == s).any()]
            print(f"   {a:<16} − {b:<16}: {ms(hieu)}   (từng seed: {', '.join(f'{h:+.3f}' for h in hieu)})")

    print("\n3. G8b(2) — khoảng cách thống kê BN giữa các máy trạm (bn_kc, trung bình 10 vòng cuối). Dự đoán: nguồn > trộn")
    for ten in ("FedBN nguồn P1", "FedBN trộn P1", "FedBN nguồn P2"):
        g = B[B.ch == ten]
        if len(g):
            print(f"   {ten:<16}: {ms(g.bn_kc)}")

    print("\n4. G8b(3) — val bAcc theo bệnh viện. Dự đoán: rosendahl lợi nhiều nhất từ FedBN ở nguồn thật")
    print(f"{'cấu hình':<17}" + "".join(f"{NGAN[n]:>13}" for n in NGUON))
    for ten in CH:
        g = B[B.ch == ten]
        if len(g):
            print(f"{ten:<17}" + "".join(f"{ms(g[NGAN[n]]):>13}" for n in NGUON))
    print("\nLưu ý: dias chỉ 74 ảnh val — chênh lệch ở dias cần KTC (kiem_tra_re.py mục 1) trước khi kết luận.")


if __name__ == "__main__":
    main()
