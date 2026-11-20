"""
phan_tich_dem14.py — đọc đêm 14 (bước 1 của thân cây chính). Không cần GPU.

    python phan_tich_dem14.py

(1) Chọn thuật toán FL nền cho ConvNeXt-atto (LayerNorm), nguồn thật, P1, ảnh 224, 3 seed:
    FedAvg (đã có) vs FedProx μ=0,01 / μ=0,1 vs FedAvgM (β=0,9, lr máy chủ 1,0).
    Luật chọn (viết TRƯỚC khi có số): chỉ thay FedAvg nếu thuật toán mới hơn FedAvg ở CẢ 3 seed
    và hơn trung bình > 0,02. Nếu không, giữ FedAvg (đơn giản hơn, dễ ghép DP hơn).
(2) Trả lời câu "LayerNorm có học nhanh / ổn định như BatchNorm không" bằng dữ liệu ĐÃ CÓ:
    độ dao động giữa seed, vòng đạt val tốt nhất, val ở vòng 10 so với vòng 30, thời gian chạy.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore", category=RuntimeWarning)


def doc(ten):
    f = RA / "lien_ket" / f"{ten}.json"
    return json.load(open(f)) if f.exists() else None


def f3(x):
    return "  —  " if x is None or np.isnan(x) else f"{x:.3f}"


def dong(nhan, tien_to, bb):
    ds = [doc(f"{tien_to}_nguon_{bb}_txlp1_kt224_s{s}") for s in range(3)]
    v = np.array([d["ket_qua"]["val"]["bacc"] if d else np.nan for d in ds])
    t = np.array([d["ket_qua"]["test"]["bacc"] if d and "test" in d["ket_qua"] else np.nan for d in ds])
    return nhan, v, t, ds


def main():
    bb = "convnext_atto"
    print("1. Thuật toán FL nền cho ConvNeXt-atto (nguồn thật, val; test để tham khảo)\n")
    goc = dong("FedAvg", "fedavg", bb)
    cac = [goc, dong("FedProx μ=0,01", "fedprox", bb), dong("FedProx μ=0,1", "fedprox_mu0.1", bb),
           dong("FedAvgM β=0,9", "fedavgm", bb)]
    print(f"{'thuật toán':<16}{'val TB':>8}{'± sd':>7}{'test TB':>9}   theo seed (val)           so FedAvg theo seed")
    for nhan, v, t, _ in cac:
        hon = v - goc[1]
        print(f"{nhan:<16}{f3(np.nanmean(v)):>8}{np.nanstd(v):>7.3f}{f3(np.nanmean(t)):>9}   "
              f"{[f3(x) for x in v]}  {['%+.3f' % x if not np.isnan(x) else '—' for x in hon]}")
    print("\n   Luật chọn: thay FedAvg chỉ khi hơn ở cả 3 seed VÀ trung bình hơn > 0,02.")
    for nhan, v, _, _ in cac[1:]:
        hon = v - goc[1]
        if np.any(np.isnan(hon)):
            print(f"   {nhan}: chưa đủ 3 seed"); continue
        ok = np.all(hon > 0) and hon.mean() > 0.02
        print(f"   {nhan}: {'ĐẠT — cân nhắc thay FedAvg' if ok else 'không đạt — giữ FedAvg'} (TB {hon.mean():+.3f})")

    print("\n2. Học nhanh / ổn định theo kiểu chuẩn hoá (FedAvg, nguồn thật, ảnh 224, dữ liệu có sẵn)\n")
    print(f"{'mô hình':<24}{'chuẩn hoá':<11}{'val TB':>7}{'sd seed':>9}{'vòng tốt nhất':>15}{'val v10':>9}{'val v30':>9}{'phút/lượt':>11}")
    for ten, cn in (("efficientnet_b0", "BatchNorm"), ("resnet18", "BatchNorm"),
                    ("convnext_atto", "LayerNorm"), ("deit_tiny_patch16_224", "LayerNorm")):
        _, v, _, ds = dong("", "fedavg", ten)
        vt, v10, v30, ph = [], [], [], []
        for d in ds:
            if not d or not d.get("lich_su"):
                continue
            ls = [r["val_bacc"] for r in d["lich_su"]]
            vt.append(int(np.argmax(ls)) + 1); v10.append(ls[min(9, len(ls) - 1)]); v30.append(ls[-1])
            ph.append(d.get("giay", np.nan) / 60)
        m = lambda a: np.mean(a) if a else np.nan
        print(f"{ten:<24}{cn:<11}{f3(np.nanmean(v)):>7}{np.nanstd(v):>9.3f}{str(vt):>15}{f3(m(v10)):>9}{f3(m(v30)):>9}{m(ph):>11.1f}")
    print("\n   Đọc: 'vòng tốt nhất' gần 30 = còn đang lên khi dừng (có thể học chưa tới);"
          "\n   val v10 sát v30 = học nhanh; sd seed nhỏ = ổn định. Phút/lượt gồm cả chấm val mỗi vòng."
          "\n   Lưu ý: hai phía khác cả kiến trúc, nên đây là mô tả, chưa phải phép thử sạch về chuẩn hoá.")


if __name__ == "__main__":
    main()
