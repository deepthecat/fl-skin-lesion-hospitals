#!/usr/bin/env bash
# Đêm 23 (~3 giờ): Fed-ISIC — ConvNeXt thua vì ít vòng? FedAvg 27 vòng × 33 bước (cùng tổng bước với 9 × 100), công thức HAM, 2 mô hình × 3 seed.
# File này tự chứa: tự viết va_isic4.py, phan_tich_isic5.py và ghi chú đêm 22 rồi mới chạy.
set -u
cd ~/ham_fl
source .venv/bin/activate
for f in fed_isic.py du_lieu.py bao_dong.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
grep -q 'cong-thuc' fed_isic.py || { echo "LỖI: fed_isic.py chưa có --cong-thuc (đêm 22 chưa vá?)"; exit 1; }
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
cat > va_isic4.py <<'PY'
"""va_isic4.py — thêm số vòng / số bước vào tên lượt fed_isic.py khi khác mặc định (để lượt 27×33 không bị bỏ qua vì 'đã có'). Chạy lại vô hại."""
from pathlib import Path
f = Path("fed_isic.py"); s = f.read_text()
cu = 'ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{ctt}{kbt}_s{arg.seed}"'
moi = ('vbt = f"_v{arg.vong}b{arg.so_buoc}" if (arg.vong or arg.so_buoc != 100) else ""\n'
       '    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{ctt}{kbt}{vbt}_s{arg.seed}"')
if "vbt =" in s:
    print("va_isic4: đã vá từ trước.")
elif cu in s:
    f.write_text(s.replace(cu, moi)); print("va_isic4: đã vá tên (thêm _v{vòng}b{bước}).")
else:
    raise SystemExit("va_isic4: KHÔNG tìm thấy dòng tên — dán lỗi này cho Claude.")
PY
cat > phan_tich_isic5.py <<'PY'
"""
phan_tich_isic5.py — Đêm 23: ConvNeXt thua trên Fed-ISIC vì ÍT VÒNG (giải thích B) hay không?
Cùng công thức HAM, cùng TỔNG số bước: 9 vòng × 100 bước (đêm 22) vs 27 vòng × 33 bước (đêm 23). bAcc test, TB đều 6 trung tâm, 3 seed.
Luật viết trước: B đúng nếu ở 27×33, ConvNeXt − EfficientNet > 0,02 VÀ dương ở cả 3 seed.
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — FedAvg, công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'lịch FL':<14}{'EfficientNet-B0':>17}{'ConvNeXt-atto':>15}{'ConvNeXt − EffNet':>19}   chênh theo seed")
    kq = {}
    for nhan, hau in [("9 × 100", ""), ("27 × 33", "_v27b33")]:
        e = doc(f"isic_fedavg_cc_efficientnet_b0_ham{hau}"); c = doc(f"isic_fedavg_cc_convnext_atto_ham{hau}")
        kq[nhan] = (e, c)
        print(f"{nhan:<14}{np.nanmean(e):>17.3f}{np.nanmean(c):>15.3f}{np.nanmean(c - e):>+19.3f}   {[f'{v:+.3f}' for v in c - e]}")
    e, c = kq["27 × 33"]; d = c - e
    print("\nTheo trung tâm, 27 × 33 (EffNet / ConvNeXt):")
    for t in TT:
        print(f"  {t:<10}{np.nanmean(doc('isic_fedavg_cc_efficientnet_b0_ham_v27b33', 'test_' + t)):.3f} / "
              f"{np.nanmean(doc('isic_fedavg_cc_convnext_atto_ham_v27b33', 'test_' + t)):.3f}")
    if np.isnan(d).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok = np.nanmean(d) > 0.02 and (d > 0).all()
    print("\nKẾT LUẬN theo luật viết trước:",
          "B ĐÚNG — ConvNeXt thua ở đêm 22 vì ít vòng; lợi thế LayerNorm lặp lại khi đủ vòng." if ok else
          "B KHÔNG được ủng hộ — lợi thế kiến trúc chỉ có trên HAM; cơ chế G8b vẫn lặp lại (đêm 21).")
    print("Tổng tập trung (đêm 22, để tính khoảng): EffNet 0,738 · ConvNeXt 0,733.")


if __name__ == "__main__":
    main()
PY
python va_isic4.py || exit 1
if ! grep -q "Kết quả đêm 22" BAN_DO.md; then cat >> BAN_DO.md <<'MD'

### Kết quả đêm 22 (23.10.2026) — Fed-ISIC2019, CÙNG công thức HAM (AdamW 3e-4 + cos), tập trung + FedAvg 9 vòng × 100 bước, bAcc test TB đều 6 trung tâm, 3 seed
| mô hình | tập trung | FedAvg | khoảng | FedAvg theo seed |
|---|---|---|---|---|
| EfficientNet-B0 (timm) | 0,738 | 0,643 | 0,095 | 0,620 / 0,649 / 0,661 |
| ConvNeXt-atto (timm) | 0,733 | 0,628 | 0,105 | 0,633 / 0,637 / 0,614 |
- ĐO ĐƯỢC: phép so hợp lệ (ConvNeXt tập trung 0,733 > 0,70). Dự đoán "khoảng ConvNeXt nhỏ hơn" SAI; hai mô hình hoà (seed chồng nhau). Trên HAM ConvNeXt hơn 0,08 → lợi thế kiến trúc CHƯA lặp lại trên bộ thứ hai. ConvNeXt thiệt nặng nhất ở dias (0,764 → 0,634).
- GIẢ THUYẾT (chưa kiểm): (A) Fed-ISIC còn ít hại do máy để gỡ (phần do máy 28%, ~0,03); (B) giao thức FLamby chỉ 9 vòng, trung tâm nhỏ học ~18 epoch mỗi vòng, trong khi trên HAM ConvNeXt còn lên tới vòng 25–29. Siomos (TMLR 2025, Table 9, 80 vòng × 200 bước) thấy ConvNeXt-S hơn EffNetV2-S → nghiêng về B.
- 23.10: tổng hợp toàn dự án (doc Claude "Tổng hợp dự án FL HAM10000 — đến đêm 22"). Chốt: đêm 23 là đêm GPU cuối trước v5; sau đó đóng băng thí nghiệm, ý mới để v6.
### Đêm 23 (23.10.2026): kiểm B — FedAvg 27 vòng × 33 bước (cùng tổng bước), công thức HAM, ConvNeXt-atto + EfficientNet-B0, 3 seed. Luật viết trước: B đúng nếu ConvNeXt − EfficientNet > 0,02 và dương ở cả 3 seed. Vá fed_isic.py: tên thêm _v{vòng}b{bước} khi khác mặc định.
MD
echo "Đã ghi đêm 22 vào BAN_DO.md."; fi
export HAMFL_GIA_THUYET="thiet_ke_1,du_lieu_thu_hai,so_vong"
export HAMFL_DU_DOAN="27x33: ConvNeXt − EfficientNet FedAvg > 0,02, dương cả 3 seed (giải thích B)"
echo ">>> Tự kiểm: FedAvg ConvNeXt 2 vòng × 3 bước, 50 ảnh/trung tâm (thư mục thử)."
python fed_isic.py --che-do fedavg --backbone convnext_atto --cong-thuc ham --vong 2 --so-buoc 3 --gioi-han-anh 50 --workers 2 --ra ketqua/fed_isic_thu > dem23_thu.txt 2>&1
rc=$?; grep -E "vòng|Đã ghi|Error|Traceback" dem23_thu.txt | tail -3
[ $rc -eq 0 ] && grep -q "_v2b3_" dem23_thu.txt || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem23_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong (tên có _v2b3 → vá đúng)."
git add -A && git commit -q -m "dem23: Fed-ISIC 27x33, kiem giai thich so vong" || true
cat > hang_doi_dem23.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do fedavg --backbone $bb --cong-thuc ham --vong 27 --so-buoc 33 --workers 2 --seed $s
done; done
echo "=== XONG ĐÊM 23 $(date) ==="
python phan_tich_isic5.py > dem23_phan_tich.txt 2>&1; cat dem23_phan_tich.txt
python bao_dong.py "Đêm 23 xong" "$(head -6 dem23_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem23.sh > dem23.txt 2>&1 &
sleep 120; echo "--- 2 phút đầu:"; grep -v Warning dem23.txt | tail -3; free -h | head -2
