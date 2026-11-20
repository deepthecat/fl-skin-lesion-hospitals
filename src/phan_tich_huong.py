"""
phan_tich_huong.py — đọc kết quả đêm 8 (G8 + G9 trong Sổ kiểm chứng). Không cần GPU, vài giây.

    python phan_tich_huong.py

G8: khác biệt máy hại FL vì cập nhật các bệnh viện kéo NGƯỢC CHIỀU nhau?
    -> so cosine giữa cập nhật các máy trạm: 4 nguồn thật vs trộn máy (trộn giữ kích thước + nhãn, chỉ xoá máy).
       Nếu đúng: trộn máy có cosine cao hơn rõ, nhất là các cặp có rosendahl.
G9: FedProx không giúp vì không có "trôi"?
    -> so độ trôi (độ dài cập nhật) và val bAcc: FedAvg vs FedProx μ = 0,1 và 1.
       Nếu μ lớn kéo độ trôi xuống mà bAcc không lên -> trôi không phải vấn đề. Nếu bAcc lên -> μ = 0,01 quá nhỏ.
"""
import json

import numpy as np

from du_lieu import RA

TEN_MAY = ["molemax", "modern", "rosendahl", "dias"]   # thứ tự máy trạm 0..3 trong kịch bản nguon
CAU_HINH = {
    "FedAvg, 4 nguồn thật": "fedavg_nguon",
    "FedAvg, trộn máy": "fedavg_tron_nguon",
    "FedProx μ=0,1, nguồn thật": "fedprox_mu0.1_nguon",
    "FedProx μ=1, nguồn thật": "fedprox_mu1_nguon",
}
GIAI_DOAN = {"vòng 1–10": slice(0, 10), "vòng 11–30": slice(10, 30)}


def doc(tien_to):
    ra = []
    for s in (0, 1, 2):
        f = RA / "lien_ket" / f"{tien_to}_efficientnet_b0_txlp1_gh_kt224_s{s}.json"
        if f.exists():
            ra.append(json.load(open(f)))
    return ra


def main():
    co = {ten: doc(tt) for ten, tt in CAU_HINH.items()}
    if not any(co.values()):
        raise SystemExit("Chưa có file *_gh_*.json nào trong ketqua/lien_ket.")

    print("0. ĐỘ CHÍNH XÁC (val / test balanced accuracy, mô hình tốt nhất trên val; trung bình seed)\n")
    print(f"{'cấu hình':<30}{'n':>2}{'val':>8}{'test':>8}")
    for ten, ds in co.items():
        if ds:
            v = [d["ket_qua"]["val"]["bacc"] for d in ds]; t = [d["ket_qua"]["test"]["bacc"] for d in ds]
            print(f"{ten:<30}{len(ds):>2}{np.mean(v):>8.3f}{np.mean(t):>8.3f}")

    print("\n1. G8 — COSINE GIỮA CẬP NHẬT CÁC MÁY TRẠM (1 = cùng chiều, 0 = vuông góc, âm = ngược chiều)")
    print("   Ở kịch bản trộn, 'máy trạm k' có cùng kích thước + nhãn với bệnh viện k nhưng ảnh từ cả 4 máy.\n")
    cap = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for gd, sl in GIAI_DOAN.items():
        print(f"   {gd}")
        print(f"{'cấu hình':<30}" + "".join(f"{TEN_MAY[i][:4]}-{TEN_MAY[j][:4]:>5}" for i, j in cap))
        for ten, ds in co.items():
            if not ds or "cos" not in ds[0]["lich_su"][0]:
                continue
            C = np.mean([[h["cos"] for h in d["lich_su"][sl]] for d in ds], axis=(0, 1))
            print(f"{ten:<30}" + "".join(f"{C[i, j]:>10.3f}" for i, j in cap))
        print()

    print("2. MỖI MÁY TRẠM CÙNG CHIỀU VỚI CẬP NHẬT TRUNG BÌNH ĐẾN ĐÂU (cos_tb), vòng 1–30")
    print("   Thấp = phần chỉnh của máy đó bị phép trung bình ghi đè nhiều.\n")
    print(f"{'cấu hình':<30}" + "".join(f"{m:>11}" for m in TEN_MAY))
    for ten, ds in co.items():
        if ds and "cos_tb" in ds[0]["lich_su"][0]:
            c = np.mean([[h["cos_tb"] for h in d["lich_su"]] for d in ds], axis=(0, 1))
            print(f"{ten:<30}" + "".join(f"{x:>11.3f}" for x in c))

    print("\n3. G9 — ĐỘ TRÔI (độ dài cập nhật mỗi vòng, trung bình vòng 1–30)\n")
    print(f"{'cấu hình':<30}" + "".join(f"{m:>11}" for m in TEN_MAY))
    for ten, ds in co.items():
        if ds and "do_troi" in ds[0]["lich_su"][0]:
            c = np.mean([[h["do_troi"] for h in d["lich_su"]] for d in ds], axis=(0, 1))
            print(f"{ten:<30}" + "".join(f"{x:>11.4f}" for x in c))
    print("\nĐối chiếu: FedAvg cũ (không _gh) nguồn thật val 0,659, trộn máy 0,707; FedProx μ=0,01 val 0,660.")
    print("Bản _gh phải ra gần như cũ — nếu lệch > 0,02 thì báo lại trước khi đọc tiếp.")


if __name__ == "__main__":
    main()
