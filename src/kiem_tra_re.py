"""
kiem_tra_re.py — GĐ1, các kiểm tra rẻ (không cần GPU, chỉ đọc logit đã lưu, ~1–3 phút).

    python kiem_tra_re.py

Mỗi mục trả lời một câu trong Sổ kiểm chứng / Bản đồ mở rộng (BAN_DO.md). P1, 3 seed.
Bootstrap theo TỔN THƯƠNG (lesion_id), không theo ảnh: các ảnh cùng một nốt không độc lập.

  1. A3/G1  Khoảng cách tập trung − FedAvg theo site, kèm KTC 95%. dias có thật sự tệ nhất không?
  2. A7/H2  AUC tấn công + TPR@1%FPR theo site và lớp, kèm KTC 95%. Đây là phép thử H2 đúng như v4 đã hứa.
  3. A2/G2  Lối tắt máy → NV? Trên ảnh val KHÔNG phải NV: tỉ lệ bị đoán NV, tách theo nguồn.
  4. A4/G3  FL nghiêng NV do nhãn hay máy? Tỉ lệ đoán NV: tập trung / nguồn thật / trộn máy.
  5. A12/G4 Lộ lớp-trong-site: AUC tấn công cho từng cặp (site, lớp) đủ ảnh.
  6. A12/G5 Hiếm ↔ lộ: số ảnh train, khoảng loss (val − train) và AUC theo lớp.
  7. A5     Val 0,78 → test 0,70: recall từng lớp trên val và test.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from du_lieu import LOP, RA
from phan_tich_theo_site import NGAN, NGUON
from tan_cong_thanh_vien import loss_tung_anh

warnings.filterwarnings("ignore")
B = 1000
SEED = (0, 1, 2)
F = {
    "tập trung": ("tap_trung", "efficientnet_b0_kt224_nghich_txlp1_s{s}"),
    "FedAvg nguồn": ("lien_ket", "fedavg_nguon_efficientnet_b0_txlp1_kt224_s{s}"),
    "FedAvg trộn": ("lien_ket", "fedavg_tron_nguon_efficientnet_b0_txlp1_kt224_s{s}"),
}
NV = LOP.index("NV")


def doc(ten, s):
    tm, mau = F[ten]
    d = np.load(RA / tm / f"{mau.format(s=s)}_logit.npz")
    return pd.DataFrame({"image": d["image"], "phan": d["phan"], "y": d["nhan"]}), d["logit"].astype(np.float64)


def ktc(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 20:
        return "—"
    return f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"


def tpr1(la, diem):
    fpr, tpr, _ = roc_curve(la, diem)
    return float(np.interp(0.01, fpr, tpr))


def chi_muc_bootstrap(nhom, rng):
    """Lấy lại các tổn thương có hoàn lại; trả về chỉ số ảnh."""
    ma, vi_tri = np.unique(nhom, return_inverse=True)
    theo = [np.flatnonzero(vi_tri == k) for k in range(len(ma))]
    chon = rng.integers(0, len(ma), len(ma))
    return np.concatenate([theo[k] for k in chon])


def main():
    cs = pd.read_csv(RA / "chia_tap.csv")
    meta = cs.set_index("image")[["nguon", "lesion_id"]]
    rng = np.random.default_rng(0)
    D = {(t, s): doc(t, s) for t in F for s in SEED}
    for (t, s), (df, lg) in D.items():
        df["nguon"] = df.image.map(meta.nguon); df["lesion"] = df.image.map(meta.lesion_id)
        df["yh"] = lg.argmax(1); df["loss"] = loss_tung_anh(lg, df.y.values)

    # ---------------------------------------------------------------- 1
    print("1. KHOẢNG CÁCH VAL bAcc (tập trung − FedAvg nguồn thật) THEO SITE, KTC 95% bootstrap theo tổn thương")
    print(f"{'site':<11}{'n val':>6}{'khoảng':>9}   KTC 95%")
    for n in NGUON:
        df0 = D[("tập trung", 0)][0]
        m = (df0.phan == "val") & (df0.nguon == n)
        idx0 = np.flatnonzero(m); les = df0.lesion.values[idx0]
        diem, bs = [], []
        for s in SEED:
            a, b = D[("tập trung", s)][0], D[("FedAvg nguồn", s)][0]
            assert (a.image.values == b.image.values).all()
            y, ya, yb = a.y.values[idx0], a.yh.values[idx0], b.yh.values[idx0]
            diem.append(balanced_accuracy_score(y, ya) - balanced_accuracy_score(y, yb))
        for _ in range(B):
            j = chi_muc_bootstrap(les, rng)
            g = []
            for s in SEED:
                a, b = D[("tập trung", s)][0], D[("FedAvg nguồn", s)][0]
                ii = idx0[j]
                g.append(balanced_accuracy_score(a.y.values[ii], a.yh.values[ii]) - balanced_accuracy_score(a.y.values[ii], b.yh.values[ii]))
            bs.append(np.mean(g))
        print(f"{NGAN[n]:<11}{len(idx0):>6}{np.mean(diem):>9.3f}   {ktc(bs)}")

    # ---------------------------------------------------------------- 2
    print("\n2. H2 — AUC TẤN CÔNG VÀ TPR@1%FPR THEO NHÓM, KTC 95% (seed 0; thành viên = train, không = val)")
    print("   H2 bị bác cho một cặp nhóm nếu KTC của chúng chồng nhau. TPR@1%FPR với nhóm nhỏ rất nhiễu.")
    for t in ("tập trung", "FedAvg nguồn"):
        df, _ = D[(t, 0)]
        ch = df[df.phan.isin(["train", "val"])].reset_index(drop=True)
        la = (ch.phan == "train").astype(int).values; diem = -ch.loss.values
        print(f"\n   {t}")
        print(f"   {'nhóm':<11}{'train':>6}{'val':>5}{'AUC':>7}   {'KTC 95% AUC':<18}{'TPR@1%':>7}")
        nhom = [(NGAN[n], ch.nguon.values == n) for n in NGUON] + [(l, ch.y.values == c) for c, l in enumerate(LOP)]
        for ten, m in nhom:
            ii = np.flatnonzero(m); l_, d_ = la[ii], diem[ii]
            if l_.sum() < 5 or (1 - l_).sum() < 5:
                continue
            les = ch.lesion.values[ii]; bs = []
            for _ in range(B):
                j = chi_muc_bootstrap(les, rng)
                if 0 < l_[j].sum() < len(j):
                    bs.append(roc_auc_score(l_[j], d_[j]))
            print(f"   {ten:<11}{l_.sum():>6}{(1 - l_).sum():>5}{roc_auc_score(l_, d_):>7.3f}   {ktc(bs):<18}{tpr1(l_, d_):>7.3f}")

    # ---------------------------------------------------------------- 3
    print("\n3. LỐI TẮT MÁY → NV? Ảnh val có bệnh thật KHÁC NV: tỉ lệ bị đoán là NV (trung bình 3 seed)")
    print("   Nếu FL học 'tông molemax = NV', ảnh không-NV của molemax sẽ bị đoán NV tăng mạnh hơn các nguồn khác.")
    print(f"{'nguồn':<11}{'n':>5}" + "".join(f"{t:>15}" for t in F))
    for n in NGUON:
        hang = []
        for t in F:
            v = [((df.phan == "val") & (df.nguon == n) & (df.y != NV)) for df, _ in [D[(t, s)] for s in SEED]]
            hang.append(np.mean([(D[(t, s)][0].yh.values[v[k].values] == NV).mean() for k, s in enumerate(SEED)]))
        so = int(v[0].sum())
        print(f"{NGAN[n]:<11}{so:>5}" + "".join(f"{x:>15.3f}" for x in hang))

    # ---------------------------------------------------------------- 4
    print("\n4. TỈ LỆ ẢNH BỊ ĐOÁN LÀ NV (thật: val 67%, test 60%) — trộn máy giữ nhãn, chỉ xoá khác biệt máy")
    print(f"{'tập':<6}" + "".join(f"{t:>15}" for t in F))
    for p in ("val", "test"):
        print(f"{p:<6}" + "".join(f"{np.mean([(D[(t, s)][0].yh.values[D[(t, s)][0].phan.values == p] == NV).mean() for s in SEED]):>15.3f}" for t in F))

    # ---------------------------------------------------------------- 5
    print("\n5. LỘ LỚP-TRONG-SITE (AUC tấn công, tập trung, trung bình 3 seed; chỉ ô có ≥ 20 train và ≥ 8 val)")
    print(f"{'lớp':<7}" + "".join(f"{NGAN[n]:>11}" for n in NGUON))
    for c, l in enumerate(LOP):
        o = []
        for n in NGUON:
            a = []
            for s in SEED:
                df = D[("tập trung", s)][0]
                m = df.phan.isin(["train", "val"]).values & (df.nguon.values == n) & (df.y.values == c)
                la = (df.phan.values[m] == "train").astype(int)
                if la.sum() >= 20 and (1 - la).sum() >= 8:
                    a.append(roc_auc_score(la, -df.loss.values[m]))
            o.append(f"{np.mean(a):>11.3f}" if a else f"{'·':>11}")
        print(f"{l:<7}" + "".join(o))

    # ---------------------------------------------------------------- 6
    print("\n6. HIẾM ↔ LỘ? Theo lớp (tập trung, trung bình 3 seed)")
    print(f"{'lớp':<7}{'n train':>8}{'loss train':>12}{'loss val':>10}{'khoảng':>9}{'AUC':>7}")
    so, kh, au = [], [], []
    for c, l in enumerate(LOP):
        r = []
        for s in SEED:
            df = D[("tập trung", s)][0]
            tr = (df.phan == "train") & (df.y == c); va = (df.phan == "val") & (df.y == c)
            m = tr | va
            r.append((tr.sum(), df.loss[tr].mean(), df.loss[va].mean(),
                      roc_auc_score(tr[m].astype(int), -df.loss[m])))
        r = np.mean(r, axis=0)
        so.append(r[0]); kh.append(r[2] - r[1]); au.append(r[3])
        print(f"{l:<7}{r[0]:>8.0f}{r[1]:>12.3f}{r[2]:>10.3f}{r[2] - r[1]:>9.3f}{r[3]:>7.3f}")
    print(f"Spearman(số ảnh, AUC) = {spearmanr(so, au)[0]:+.2f}   Spearman(khoảng loss, AUC) = {spearmanr(kh, au)[0]:+.2f}   (7 lớp: chỉ để tham khảo)")

    # ---------------------------------------------------------------- 7
    print("\n7. VAL → TEST: recall từng lớp (trung bình 3 seed)")
    for t in ("tập trung", "FedAvg nguồn"):
        print(f"   {t}")
        print(f"   {'':<6}" + "".join(f"{l:>7}" for l in LOP) + f"{'bAcc':>7}")
        for p in ("val", "test"):
            rc = np.mean([[(D[(t, s)][0].yh.values[(D[(t, s)][0].phan.values == p) & (D[(t, s)][0].y.values == c)] == c).mean()
                           for c in range(len(LOP))] for s in SEED], axis=0)
            print(f"   {p:<6}" + "".join(f"{x:>7.3f}" for x in rc) + f"{rc.mean():>7.3f}")
        cnt = D[(t, 0)][0]
        print(f"   {'n val':<6}" + "".join(f"{((cnt.phan == 'val') & (cnt.y == c)).sum():>7}" for c in range(len(LOP))))
        print(f"   {'n test':<6}" + "".join(f"{((cnt.phan == 'test') & (cnt.y == c)).sum():>7}" for c in range(len(LOP))))


if __name__ == "__main__":
    main()
