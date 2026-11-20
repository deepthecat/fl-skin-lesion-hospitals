"""
phan_tich_dp_sau.py — bước 3 thân cây chính: đường đánh đổi HIỆU NĂNG ↔ RIÊNG TƯ khi áp DP ở nhiều độ sâu. Không cần GPU.

    python phan_tich_dp_sau.py

Đọc hai loại kết quả (kịch bản nguồn thật, P1, ảnh 224, 3 seed):
  L0 = dp_tuyen_tinh.py : xương sống ĐÓNG BĂNG (trọng số ImageNet), DP chỉ trên lớp phân loại cuối.
  L1, L2 = dp_sau.py    : DP-SGD trên khối cuối (L1) / hai khối cuối (L2) + đầu phân loại, phần còn lại đóng băng.
Với mỗi (mô hình, độ sâu, ε): bAcc val/test, ε thật lớn nhất trong các bệnh viện, mức rò rỉ đo bằng tấn công loss
và RMIA (dùng 8 mô hình tham chiếu của đêm 15; hệ số a chỉnh trên mô hình tham chiếu như rmia.py).
CỔNG A (viết từ 20.10): ở CÙNG ε, nếu ConvNeXt không đạt bAcc cao hơn EfficientNet-DP-lớp-cuối (L0)
→ giả thuyết "LayerNorm nối được hai trục" sai.
"""
import glob
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from du_lieu import RA, doc_chia_tap
from rmia import chi_so, diem_rmia, p_dung

warnings.filterwarnings("ignore")
MUC = ("inf", "clip", "8", "3", "1", "0.5")


def nap_tham_chieu(bb, ds, chon, la_tv):
    fs = sorted(glob.glob(str(RA / "tham_chieu" / f"tc_{bb}_txlp1_kt224_c*n*.npz")))
    if len(fs) < 2:
        return None
    R = [np.load(f) for f in fs]
    y = ds.nhan.values
    P = np.stack([p_dung(r["logit"], y) for r in R])[:, chon]
    T = np.stack([r["trong"] for r in R])[:, chon]
    gia = T[0] | (la_tv == 0)
    luoi = np.round(np.arange(0, 1.01, 0.1), 1)
    auc = [roc_auc_score(T[0][gia].astype(int), diem_rmia(P[0], P[1:], T[1:], a)[gia]) for a in luoi]
    return P, T, float(luoi[int(np.argmax(auc))])


def main():
    ds = doc_chia_tap()
    phan = ds.phan.values
    chon = np.isin(phan, ["train", "val"])
    la_tv = (phan[chon] == "train").astype(int)
    ref = {bb: nap_tham_chieu(bb, ds, chon, la_tv) for bb in ("convnext_atto", "efficientnet_b0")}
    hang = []
    mau = [("L0 đóng băng", "dp_tuyen_tinh", "dptt_nguon_eps{m}_{bb}_txlp1_kt224_s{s}")]
    mau += [(f"L0 lr={lr}", f"dp_tuyen_tinh_lr{lr}", "dptt_nguon_eps{m}_{bb}_txlp1_kt224_s{s}") for lr in ("1", "4")]
    mau += [(f"L{k} khối cuối", "dp_sau", f"dpsau_nguon_L{k}_eps{{m}}_{{bb}}_txlp1_kt224_s{{s}}") for k in (1, 2)]
    mau += [("L1 lô1024 lr3e-4", "dp_sau", "dpsau_nguon_L1_eps{m}_{bb}_txlp1_b1024_lr0.0003_kt224_s{s}")]
    for bb in ("convnext_atto", "efficientnet_b0"):
        for do_sau, thu, m_ten in mau:
            for m in MUC:
                for s in range(3):
                    f = RA / thu / (m_ten.format(m=m, bb=bb, s=s) + ".json")
                    if not f.exists():
                        continue
                    d = json.load(open(f))
                    eps = [v.get("eps_that") for v in d.get("may_tram", {}).values()]
                    eps = [float(e) for e in eps if e is not None]
                    r = dict(mo_hinh=bb, do_sau=do_sau, eps=m, seed=s, val=d["ket_qua"]["val"]["bacc"],
                             test=d["ket_qua"]["test"]["bacc"], eps_that_max=max(eps) if eps else np.nan)
                    fz = f.with_name(f.name.replace(".json", "_logit.npz"))
                    if fz.exists():
                        z = np.load(fz)
                        p = p_dung(z["logit"], ds.nhan.values)[chon]
                        r["AUC_loss"], r["TPR1_loss"] = chi_so(np.log(p + 1e-12), la_tv)
                        if ref[bb] is not None:
                            P, T, a = ref[bb]
                            r["AUC_rmia"], r["TPR1_rmia"] = chi_so(diem_rmia(p, P, T, a), la_tv)
                    hang.append(r)
    if not hang:
        raise SystemExit("Chưa có kết quả DP nào.")
    b = pd.DataFrame(hang)
    b.to_csv(RA / "dp_sau_tong_hop.csv", index=False)
    cot = [c for c in ("val", "test", "eps_that_max", "AUC_loss", "AUC_rmia", "TPR1_rmia") if c in b.columns]
    g = b.groupby(["mo_hinh", "do_sau", "eps"], sort=False)
    tb = g[cot].mean()
    tb["sd_val"] = g["val"].std()
    tb["n_seed"] = g["val"].count()
    pd.set_option("display.width", 220)
    print("Đường đánh đổi (trung bình 3 seed; nguồn thật):\n")
    print(tb.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nCỔNG A — cùng ε, ConvNeXt (mọi độ sâu/công thức, tốt nhất theo val) so với EfficientNet L0 (công thức L0 tốt nhất, val bAcc):")
    for m in ("8", "3", "1"):
        e = b[(b.mo_hinh == "efficientnet_b0") & (b.do_sau.str.startswith("L0")) & (b.eps == m)]
        e = e[e.do_sau == e.groupby("do_sau").val.mean().idxmax()] if not e.empty else e
        c = b[(b.mo_hinh == "convnext_atto") & (b.eps == m)]
        if e.empty or c.empty:
            print(f"   ε={m}: chưa đủ dữ liệu"); continue
        c_tot = c.groupby("do_sau").val.mean()
        print(f"   ε={m}: EfficientNet L0 {e.val.mean():.3f} | ConvNeXt tốt nhất {c_tot.max():.3f} ({c_tot.idxmax()})"
              f" → {'ConvNeXt hơn' if c_tot.max() > e.val.mean() else 'ConvNeXt KHÔNG hơn'}")
    print("\nĐọc: 'clip' = chỉ cắt gradient, không nhiễu (tách giá của cắt khỏi giá của nhiễu). ε thật = lớn nhất trong 4 bệnh viện.")
    print("Rò rỉ: AUC 0,5 = kín; RMIA là cận dưới chặt hơn loss. Tất cả chỉ là MÔ TẢ cho tới khi đủ 3 seed mỗi ô.")
    print(f"Đã ghi {RA / 'dp_sau_tong_hop.csv'}")


if __name__ == "__main__":
    main()
