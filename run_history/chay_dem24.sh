#!/usr/bin/env bash
# Lượt 24 (~7–8 giờ, chạy ban ngày): Fed-ISIC — ConvNeXt có thắng khi FL học LÂU HƠN (27 vòng × 100 bước = gấp 3 tổng bước)? 2 mô hình × 3 seed.
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in fed_isic.py du_lieu.py bao_dong.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
grep -q 'vbt =' fed_isic.py || { echo "LỖI: fed_isic.py chưa có bản vá đêm 23 (va_isic4)"; exit 1; }
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
if ! grep -q "Kết quả đêm 23" BAN_DO.md && [ -f "$D/BAN_DO_dem23.md" ]; then cat "$D/BAN_DO_dem23.md" >> BAN_DO.md && echo "Đã ghi đêm 23 vào BAN_DO.md."; fi
cat > phan_tich_isic6.py <<'PY'
"""
phan_tich_isic6.py — Lượt 24: ConvNeXt có vượt EfficientNet trên Fed-ISIC khi huấn luyện FL DÀI HƠN (gấp 3 tổng bước)?
FedAvg, công thức HAM, bAcc test TB đều 6 trung tâm, 3 seed. So 3 lịch: 9×100 (đêm 22), 27×33 (đêm 23, cùng tổng bước), 27×100 (gấp 3 tổng bước).
Luật viết trước: "ConvNeXt thắng khi học lâu hơn" đúng nếu ở 27×100, ConvNeXt − EfficientNet > 0,02 VÀ dương ở cả 3 seed.
Lưu ý: 27×100 học nhiều bước hơn mô hình tập trung (20 epoch) → chỉ so HAI KIẾN TRÚC với nhau, không tính khoảng tập trung − FL.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
LICH = [("9 × 100", ""), ("27 × 33", "_v27b33"), ("27 × 100 (×3 bước)", "_v27b100")]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — FedAvg, công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'lịch FL':<22}{'EfficientNet-B0':>17}{'ConvNeXt-atto':>15}{'ConvNeXt − EffNet':>19}   chênh theo seed")
    for nhan, hau in LICH:
        e = doc(f"isic_fedavg_cc_efficientnet_b0_ham{hau}"); c = doc(f"isic_fedavg_cc_convnext_atto_ham{hau}")
        print(f"{nhan:<22}{np.nanmean(e):>17.3f}{np.nanmean(c):>15.3f}{np.nanmean(c - e):>+19.3f}   {[f'{v:+.3f}' for v in c - e]}")
    h = "_v27b100"
    print("\nTheo trung tâm, 27 × 100 (EffNet / ConvNeXt):")
    for t in TT:
        print(f"  {t:<10}{np.nanmean(doc('isic_fedavg_cc_efficientnet_b0_ham' + h, 'test_' + t)):.3f} / "
              f"{np.nanmean(doc('isic_fedavg_cc_convnext_atto_ham' + h, 'test_' + t)):.3f}")
    d = doc("isic_fedavg_cc_convnext_atto_ham" + h) - doc("isic_fedavg_cc_efficientnet_b0_ham" + h)
    if np.isnan(d).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok = d.mean() > 0.02 and (d > 0).all()
    print("\nKẾT LUẬN theo luật viết trước:",
          "ĐÚNG — lợi thế ConvNeXt xuất hiện khi học FL lâu hơn (khớp Siomos TMLR 2025)." if ok else
          "KHÔNG được ủng hộ — học lâu gấp 3 vẫn hoà; lợi thế kiến trúc chỉ có trên HAM ở ngân sách này.")


if __name__ == "__main__":
    main()
PY
grep -q "Lượt 24" BAN_DO.md || cat >> BAN_DO.md <<'MD'
### Lượt 24 (26.10.2026, ban ngày): Fed-ISIC — FedAvg 27 vòng × 100 bước (GẤP 3 tổng bước), công thức HAM, ConvNeXt-atto + EfficientNet-B0, 3 seed. Kiểm giới hạn còn mở của đêm 23 ("ConvNeXt thắng khi học lâu hơn", Siomos TMLR 2025). Luật viết trước: đúng nếu ConvNeXt − EfficientNet > 0,02 và dương ở cả 3 seed. Chỉ so hai kiến trúc với nhau (FL học nhiều bước hơn tập trung).
MD
export HAMFL_GIA_THUYET="thiet_ke_1,du_lieu_thu_hai,hoc_lau"
export HAMFL_DU_DOAN="27x100: ConvNeXt − EfficientNet FedAvg > 0,02, dương cả 3 seed"
git add -A && git commit -q -m "luot24: Fed-ISIC 27x100, kiem hoc lau" || true
cat > hang_doi_dem24.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do fedavg --backbone $bb --cong-thuc ham --vong 27 --so-buoc 100 --workers 2 --seed $s
done; done
echo "=== XONG LƯỢT 24 $(date) ==="
python phan_tich_isic6.py > dem24_phan_tich.txt 2>&1; cat dem24_phan_tich.txt
cp dem24_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Lượt 24 xong" "$(head -7 dem24_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem24.sh > dem24.txt 2>&1 &
sleep 150; echo "--- 2,5 phút đầu:"; grep -v Warning dem24.txt | grep -E "vòng|fedavg|Error|Traceback" | tail -3; free -h | head -2
echo ">>> Nếu thấy dòng 'fedavg: 6 trung tâm ... 27 vòng × 100 bước' hoặc 'vòng 1/27' là ổn — đóng VS Code được."
