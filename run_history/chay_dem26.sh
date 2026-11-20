#!/usr/bin/env bash
# Lượt 26 (~3,5 giờ, chạy qua đêm): Fed-ISIC — trộn máy ở ngân sách dài (FedAvg 27×100, công thức HAM), EfficientNet-B0 + ConvNeXt-atto, 3 seed.
set -u
cd ~/ham_fl
source .venv/bin/activate
for f in fed_isic.py du_lieu.py bao_dong.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
grep -q '_e{arg.epoch}' fed_isic.py || { echo "LỖI: thiếu bản vá lượt 25"; exit 1; }
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
cat > phan_tich_isic8.py <<'PY'
"""
phan_tich_isic8.py — Lượt 26: phép trộn máy (G8b) trên Fed-ISIC ở CÙNG ngân sách dài với phép so kiến trúc (FedAvg 27 vòng × 100 bước, công thức HAM).
Luật viết trước: G8b lặp lại sạch nếu EfficientNet (BN): trộn − nguồn > 0,03 và dương cả 3 seed; ConvNeXt (LN): |trộn − nguồn| < 0,02.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
MH = [("EfficientNet-B0 (BN)", "efficientnet_b0"), ("ConvNeXt-atto (LN)", "convnext_atto")]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — FedAvg 27×100, công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'mô hình':<22}{'TT 60ep':>9}{'nguồn':>8}{'trộn':>8}{'trộn − nguồn':>14}{'phần do máy':>13}   trộn − nguồn theo seed")
    d = {}
    for nhan, bb in MH:
        t = doc(f"isic_tap_trung_cc_{bb}_ham_e60"); n = doc(f"isic_fedavg_cc_{bb}_ham_v27b100"); m = doc(f"isic_fedavg_cc_{bb}_ham_tron_v27b100")
        d[bb] = m - n; kc = np.nanmean(t - n)
        print(f"{nhan:<22}{np.nanmean(t):>9.3f}{np.nanmean(n):>8.3f}{np.nanmean(m):>8.3f}{np.nanmean(d[bb]):>+14.3f}{np.nanmean(d[bb]) / kc:>12.0%}   {[f'{v:+.3f}' for v in d[bb]]}")
    print("\nTheo trung tâm (nguồn → trộn):")
    for nhan, bb in MH:
        print(f"  {nhan}: " + "  ".join(f"{t} {np.nanmean(doc(f'isic_fedavg_cc_{bb}_ham_v27b100', 'test_' + t)):.2f}→{np.nanmean(doc(f'isic_fedavg_cc_{bb}_ham_tron_v27b100', 'test_' + t)):.2f}" for t in TT))
    e, c = d["efficientnet_b0"], d["convnext_atto"]
    if np.isnan(e).any() or np.isnan(c).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok_e = e.mean() > 0.03 and (e > 0).all(); ok_c = abs(c.mean()) < 0.02
    print(f"\nKẾT LUẬN theo luật viết trước: EfficientNet {'ĐẠT' if ok_e else 'KHÔNG đạt'} (> 0,03, cả 3 seed) · ConvNeXt {'ĐẠT' if ok_c else 'KHÔNG đạt'} (|.| < 0,02)")
    print("→ G8b lặp lại SẠCH trên Fed-ISIC ở ngân sách dài." if ok_e and ok_c else "→ G8b chỉ lặp lại một phần ở ngân sách dài — xem từng vế.")


if __name__ == "__main__":
    main()
PY
grep -q "Kết quả lượt 25" BAN_DO.md || cat >> BAN_DO.md <<'MD'
### Kết quả lượt 25 (26.10.2026) — Fed-ISIC, tập trung 60 epoch (≈ cùng tổng bước với FedAvg 27×100), công thức HAM, bAcc test TB đều 6 trung tâm, 3 seed
| mô hình | TT 20 ep | TT 60 ep | FedAvg 27×100 | khoảng (60 ep) | khoảng theo seed |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,738 | 0,777 | 0,640 | 0,137 | 0,173 / 0,101 / 0,136 |
| ConvNeXt-atto | 0,733 | 0,736 | 0,683 | 0,053 | 0,054 / 0,053 / 0,051 |
- ĐO ĐƯỢC: theo luật viết trước ĐÚNG — khoảng EffNet − khoảng ConvNeXt +0,084 (+0,119 / +0,048 / +0,085). Thiết kế 1 lặp lại trên bộ thứ hai khi cùng ngân sách dài. Cùng chiều với HAM (khoảng 0,117 vs 0,072).
- ĐO ĐƯỢC, cần nói rõ trong paper: hai nửa khác nhau. Học lâu giúp EfficientNet TẬP TRUNG (+0,039) mà không giúp FedAvg; giúp ConvNeXt FedAvg (+0,055) mà không giúp tập trung (+0,003). Tập trung 60 ep: EfficientNet > ConvNeXt (0,777 vs 0,736; modern 0,858 vs 0,681). → khoảng nhỏ của ConvNeXt là do FL tốt lên VÀ tập trung không lên; ở số tuyệt đối FedAvg ConvNeXt vẫn tốt nhất (0,683).
- GIẢ THUYẾT (chưa kiểm): EfficientNet FL kịch trần vì thống kê BN trung bình không khớp ai (G8b) — học thêm không gỡ được; ConvNeXt không có nút thắt đó.
### Lượt 26 (26–27.10.2026): trộn máy ở ngân sách dài — FedAvg 27×100 kịch bản trộn, EfficientNet + ConvNeXt, 3 seed. Luật viết trước: G8b lặp lại sạch nếu EfficientNet trộn − nguồn > 0,03 cả 3 seed VÀ ConvNeXt |trộn − nguồn| < 0,02.
MD
export HAMFL_GIA_THUYET="G8b,du_lieu_thu_hai,hoc_lau"
export HAMFL_DU_DOAN="27x100: EffNet trộn − nguồn > 0,03 cả 3 seed; ConvNeXt |trộn − nguồn| < 0,02"
echo ">>> Tự kiểm: FedAvg trộn ConvNeXt 1 vòng × 3 bước, 50 ảnh/trung tâm (thư mục thử)."
python fed_isic.py --che-do fedavg --backbone convnext_atto --cong-thuc ham --kich-ban tron --vong 1 --so-buoc 3 --gioi-han-anh 50 --workers 2 --ra ketqua/fed_isic_thu > dem26_thu.txt 2>&1
rc=$?; grep -E "vòng|Đã ghi|Error|Traceback" dem26_thu.txt | tail -2
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem26_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong."
git add -A && git commit -q -m "luot26: Fed-ISIC tron may 27x100" || true
cat > hang_doi_dem26.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in efficientnet_b0 convnext_atto; do
  python fed_isic.py --che-do fedavg --backbone $bb --cong-thuc ham --kich-ban tron --vong 27 --so-buoc 100 --workers 2 --seed $s
done; done
echo "=== XONG LƯỢT 26 $(date) ==="
python phan_tich_isic8.py > dem26_phan_tich.txt 2>&1; cat dem26_phan_tich.txt
cp dem26_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Lượt 26 xong" "$(head -6 dem26_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem26.sh > dem26.txt 2>&1 &
sleep 150; echo "--- 2,5 phút đầu:"; grep -v Warning dem26.txt | grep -E "vòng|Error|Traceback" | tail -2; free -h | head -2
echo ">>> Thấy 'vòng  1/27' là ổn — đi ngủ được."
