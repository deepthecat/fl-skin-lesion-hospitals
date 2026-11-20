"""
phan_tich_isic.py — đọc kết quả benchmark Fed-ISIC2019 và đặt cạnh số đã công bố. Không cần GPU, vài giây.

    python phan_tich_isic.py

Số công bố để đối chiếu (balanced accuracy trên test, EfficientNet-B0, giao thức FLamby):
  tập trung 0,683 / FedAvg 0,644 — lần chạy lại của Tavakoli và c.s. (MLHC 2024), đã trích trong report v4.
Hai cách tính được in ra: (a) gộp toàn bộ test, (b) trung bình bAcc từng trung tâm. Cần đối chiếu cách bảng
công bố tính trước khi kết luận "khớp" hay "lệch".
"""
import json

import numpy as np

from du_lieu import RA

TEN_TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
CHE_DO = ["tap_trung", "fedavg", "fedprox", "fedbn"]
CONG_BO = {"tap_trung": 0.683, "fedavg": 0.644}


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f}±{x.std(ddof=1):.3f}" if len(x) > 1 else (f"{x.mean():.3f}" if len(x) else "—")


def main():
    D = {}
    for m in CHE_DO:
        ds = [json.load(open(f)) for f in sorted((RA / "fed_isic").glob(f"isic_{m}_cc_s*.json"))]
        if ds:
            D[m] = ds
    if not D:
        raise SystemExit("Chưa có kết quả trong ketqua/fed_isic/")

    print("1. BALANCED ACCURACY TRÊN TEST (trung bình ± sd qua seed)\n")
    print(f"{'chế độ':<11}{'n':>2}{'gộp test':>14}{'TB theo trạm':>14}{'công bố':>9}{'phút/lượt':>10}")
    for m, ds in D.items():
        k = [d["ket_qua"] for d in ds]
        print(f"{m:<11}{len(ds):>2}{ms([x['test_gop'] for x in k]):>14}{ms([x['test_tb_tram'] for x in k]):>14}"
              f"{CONG_BO.get(m, float('nan')):>9.3f}{np.mean([d['giay'] for d in ds]) / 60:>10.1f}")
    if "tap_trung" in D and "fedavg" in D:
        for t in ("test_gop", "test_tb_tram"):
            a = np.mean([d["ket_qua"][t] for d in D["tap_trung"]]); b = np.mean([d["ket_qua"][t] for d in D["fedavg"]])
            print(f"   khoảng tập trung − FedAvg ({t}): {a - b:.3f}   (công bố: {0.683 - 0.644:.3f}; HAM10000 4 nguồn của mình: 0,119)")

    print("\n2. THEO TRUNG TÂM (test bAcc)\n")
    print(f"{'chế độ':<11}" + "".join(f"{t:>12}" for t in TEN_TT))
    for m, ds in D.items():
        print(f"{m:<11}" + "".join(f"{ms([d['ket_qua'][f'test_{t}'] for d in ds]):>12}" for t in TEN_TT))
    if "fedbn" in D:
        print(f"{'fedbn BN tb':<11}" + "".join(f"{ms([d['ket_qua'][f'bn_tb_test_{t}'] for d in D['fedbn']]):>12}" for t in TEN_TT))
        print("   (dòng cuối: FedBN nhưng dùng BN trung bình như FedAvg — để tách phần lợi do BN riêng)")
    print("\nSố ảnh test từng trung tâm: BCN 2483, molemax 791, modern 672, rosendahl 452, MSK 164, dias 88.")

    # ---------------- 3. đối chiếu với số gốc của FLamby (file kết quả trong repo FLamby, 5 seed)
    import pandas as pd
    from pathlib import Path
    thu = Path.home() / "FLamby" / "flamby" / "results" / "results_benchmark_fed_isic2019"
    tep = sorted(thu.glob("*.csv"))
    if not tep:
        return
    F = pd.concat([pd.read_csv(f) for f in tep], ignore_index=True)
    F = F[F["Test"].astype(str).str.startswith("client_test_")].copy()
    F["tram"] = F["Test"].str.replace("client_test_", "").astype(int)
    tb = F.groupby(["Method", "seed"])["Metric"].mean().groupby("Method").agg(["mean", "std", "count"])
    theo = F.groupby(["Method", "tram"])["Metric"].mean().unstack()
    print(f"\n3. SỐ GỐC CỦA FLAMBY ({len(tep)} seed) — trung bình đều bAcc 6 trung tâm, như cách FLamby báo cáo\n")
    print(f"{'phương pháp':<22}{'TB theo trạm':>14}" + "".join(f"{t:>10}" for t in TEN_TT))
    for m in tb.sort_values("mean", ascending=False).index:
        print(f"{m:<22}{tb.loc[m, 'mean']:>8.3f}±{tb.loc[m, 'std']:.3f}" + "".join(f"{theo.loc[m, k]:>10.3f}" for k in range(6)))
    print("\nSO VỚI MÌNH (cột 'TB theo trạm' ở mục 1):")
    for ta, fl in (("tap_trung", "Pooled Training"), ("fedavg", "FedAvg"), ("fedprox", "FedProx")):
        if ta in D and fl in tb.index:
            x = np.mean([d["ket_qua"]["test_tb_tram"] for d in D[ta]])
            print(f"   {ta:<10} mình {x:.3f}   FLamby {tb.loc[fl, 'mean']:.3f}   chênh {x - tb.loc[fl, 'mean']:+.3f}")


if __name__ == "__main__":
    main()
