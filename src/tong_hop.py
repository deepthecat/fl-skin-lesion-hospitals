"""
tong_hop.py — gộp mọi kết quả thành bảng trung bình ± độ lệch chuẩn theo seed.

    python tan_cong_thanh_vien.py && python hieu_chinh_nguong.py && python tong_hop.py

Đọc: ketqua/tap_trung/*.json, ketqua/lien_ket/*.json, và hai file CSV của tấn công + hiệu chỉnh.
Báo luôn lượt nào trong kế hoạch còn thiếu, lượt nào đang dở (còn checkpoint).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from du_lieu import RA

KE_HOACH = ([("fedavg", "mot_may", "efficientnet_b0", 0)]
            + [("fedavg", kb, "efficientnet_b0", s) for s in (0, 1, 2) for kb in ("iid", "dir1.0", "dir0.3", "dir0.1")]
            + [("fedavg", "iid_anh", "efficientnet_b0", 0)]
            + [(tt, kb, "efficientnet_b0", 0) for tt in ("fedprox", "fedavgm") for kb in ("dir0.3", "dir0.1")]
            + [("fedavg", kb, "resnet18", 0) for kb in ("iid", "dir0.3", "dir0.1")])


KE_HOACH_DP = [(kb, e) for kb in ("mot_may", "iid") for e in ("inf", "8", "3", "1")]


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}        "


def doc(thu_muc, loai):
    hang = []
    for f in sorted(glob.glob(str(RA / thu_muc / "*.json"))):
        j = json.load(open(f))
        c, k = j["cau_hinh"], j["ket_qua"]
        tt = {"dp": f"DP toàn mạng ε={j.get('epsilon')}",
              "dp_tt": f"DP tuyến tính ε={j.get('epsilon')}"}.get(loai, c.get("thuat_toan", "tập trung"))
        if c.get("tien_xu_ly", "p0") != "p0":
            tt = f"{tt} [txl:{c['tien_xu_ly']}]"
        if c.get("trong_so", "nghich") != "nghich":
            tt = f"{tt} [w:{c['trong_so']}]"
        if c.get("trong_so_cuc_bo"):
            tt = f"{tt} [trọng số cục bộ]"
        if c.get("dong_bang", "khong") != "khong":
            tt = f"{tt} [đb:{c['dong_bang']}]"
        hang.append(dict(ten=Path(f).stem, loai=loai, thuat_toan=tt,
                         kich_ban=c.get("kich_ban", "—"), backbone=c["backbone"], seed=c["seed"],
                         val=k["val"]["bacc"], test=k["test"]["bacc"], f1=k["test"]["f1_macro"],
                         auc=k["test"]["auc_macro"]))
    return pd.DataFrame(hang)


def main():
    ds = pd.concat([doc("tap_trung", "tap_trung"), doc("lien_ket", "lien_ket"), doc("dp", "dp"),
                    doc("dp_tuyen_tinh", "dp_tt")], ignore_index=True)

    # ---- tiến độ
    co = {(r.thuat_toan, r.kich_ban, r.backbone, r.seed) for r in ds.itertuples() if r.loai == "lien_ket"}
    thieu = [k for k in KE_HOACH if k not in co]
    do_dang = sorted(Path(p).name for p in glob.glob(str(RA / "lien_ket" / "*_ckpt.pt")))
    print(f"TIẾN ĐỘ federated: {len(KE_HOACH) - len(thieu)} / {len(KE_HOACH)} lượt trong kế hoạch")
    for k in thieu:
        print("   thiếu:", " | ".join(map(str, k)))
    for f in do_dang:
        print("   đang dở (còn checkpoint):", f)
    co_dp = {(r.kich_ban, r.thuat_toan.split("=")[-1]) for r in ds.itertuples() if r.loai == "dp"}
    thieu_dp = [k for k in KE_HOACH_DP if k not in co_dp]
    print(f"TIẾN ĐỘ DP: {len(KE_HOACH_DP) - len(thieu_dp)} / {len(KE_HOACH_DP)}")
    for k in thieu_dp:
        print("   thiếu: DP |", " | ε=".join(k))
    for f in sorted(Path(p).name for p in glob.glob(str(RA / "dp" / "*_ckpt.pt"))):
        print("   đang dở (còn checkpoint):", f)

    # ---- gộp thêm tấn công + hiệu chỉnh nếu có
    for tep, cot in (("tan_cong_thanh_vien.csv", ["AUC", "khoang_cach"]),
                     ("hieu_chinh_nguong.csv", ["lay_lai_A", "lay_lai_C", "C_7_do_lech", "ti_le_doan_NV"])):
        f = RA / tep
        if f.exists():
            ds = ds.merge(pd.read_csv(f)[["mo_hinh"] + cot].rename(columns={"mo_hinh": "ten"}), on="ten", how="left")

    nhom = ["loai", "thuat_toan", "kich_ban", "backbone"]
    thu_tu = {"—": 0, "mot_may": 1, "iid": 2, "iid_anh": 3, "nguon": 4, "dir1.0": 5, "dir0.3": 6, "dir0.1": 7}
    ds["_tt"] = ds.kich_ban.map(thu_tu)
    print("\nBẢNG GỘP (trung bình ± độ lệch chuẩn qua các seed; n = số seed)")
    cot_in = [("test", "test bacc"), ("val", "val bacc"), ("auc", "test AUC")]
    cot_in += [(c, c) for c in ("AUC", "khoang_cach", "lay_lai_C", "C_7_do_lech") if c in ds.columns]
    tieu_de = f"{'nhóm':<44}{'n':>2}  " + "".join(f"{t:>18}" for _, t in cot_in)
    print(tieu_de)
    for khoa, g in ds.sort_values(["loai", "_tt", "thuat_toan", "backbone"]).groupby(nhom, sort=False):
        ten = " | ".join(khoa[1:])
        print(f"{ten:<44}{len(g):>2}  " + "".join(f"{ms(g[c].dropna()) if g[c].notna().any() else '—':>18}"
                                                  for c, _ in cot_in))
    print("\nChú thích: 'AUC' = AUC tấn công suy luận thành viên (0.5 = kín); 'test AUC' = AUC phân loại.")
    print("'lay_lai_C' = bacc lấy lại bằng hiệu chỉnh 7 độ lệch trên val; 'C_7_do_lech' = test bacc sau hiệu chỉnh.")
    ds.drop(columns="_tt").to_csv(RA / "tong_hop.csv", index=False)
    print(f"Đã ghi {RA / 'tong_hop.csv'}")


if __name__ == "__main__":
    main()
