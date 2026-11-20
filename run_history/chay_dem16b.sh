#!/usr/bin/env bash
# Đêm 16b — bước 3 thân cây chính: DP-SGD ở khối cuối (L1) và hai khối cuối (L2), FL thật, nguồn thật, P1, 3 seed.
#   Ưu tiên (chạy theo thứ tự, dừng giữa chừng vẫn có kết quả dùng được):
#   L1 × {3, inf, 1, 8, clip} × {ConvNeXt, EfficientNet} × seed 0,1,2  → rồi  L2 × {3, inf} × 2 mô hình × 3 seed
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in dp_sau.py dp_tuyen_tinh.py phan_tich_dp_sau.py rmia.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
python -c "import torch; from torch.func import vmap, grad, functional_call; import opacus" || { echo "LỖI: thiếu torch.func hoặc opacus"; exit 1; }
if pgrep -f hang_doi_dem16a.sh >/dev/null; then
  echo ">>> Đêm 16a còn đang chạy — đợi nó xong rồi mới tự kiểm (tránh tranh GPU 4 GB)..."
  while pgrep -f hang_doi_dem16a.sh >/dev/null; do sleep 30; done
fi
pgrep -f "hang_doi_dem1[0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }

# ---------- 0. Ghi chép BAN_DO.md
F15=$(ls -t "$D"/BAN_DO_dem15*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 15" BAN_DO.md && [ -n "$F15" ]; then cat "$F15" >> BAN_DO.md && echo "Đã ghi đêm 15 + kế hoạch đêm 16."; fi

export HAMFL_GIA_THUYET="thiet_ke_B3,cong_A"
export HAMFL_DU_DOAN="Cổng A: ở ε=3 và ε=1, ConvNeXt (L1 hoặc L2) có val bAcc cao hơn EfficientNet L0; DP sâu hơn (L1) tốt hơn L0 khi ε≥3, kém hơn khi ε=1"

# ---------- 1. Tự kiểm (~5–8 phút, im lặng giữa các dòng)
echo ">>> Tự kiểm 1/3: ConvNeXt L1 không DP, 2 vòng — có học được không + tốc độ."
python dp_sau.py --backbone convnext_atto --muc-mo 1 --epsilon inf --seed 0 --dung-sau-vong 2 > dem16b_thu1.txt 2>&1
rc=$?; grep -E "tham số|vòng|dừng thử|Error|Traceback" dem16b_thu1.txt | grep -v Warning | tail -4
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm 1 hỏng (mã $rc) — dán dem16b_thu1.txt cho Claude"; exit 1; }
v2=$(grep -oP 'vòng +2/30 +val bacc \K[0-9.]+' dem16b_thu1.txt)
awk "BEGIN{exit !($v2 > 0.25)}" || { echo "LỖI: sau 2 vòng val bAcc chỉ $v2 (≈ đoán mò) — có thể lr sai. Dán dem16b_thu1.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm 2/3: ConvNeXt L2 có DP, 1 vòng — bộ nhớ + tốc độ (cấu hình nặng nhất)."
python dp_sau.py --backbone convnext_atto --muc-mo 2 --epsilon 3 --seed 0 --dung-sau-vong 1 > dem16b_thu2.txt 2>&1
rc=$?; grep -E "tham số|ε thật|dừng thử|Error|Traceback|out of memory" dem16b_thu2.txt | tail -6
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm 2 hỏng (mã $rc) — dán dem16b_thu2.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm 3/3: EfficientNet L1 có DP, 1 vòng."
python dp_sau.py --backbone efficientnet_b0 --muc-mo 1 --epsilon 3 --seed 0 --dung-sau-vong 1 > dem16b_thu3.txt 2>&1
rc=$?; grep -E "tham số|dừng thử|Error|Traceback|out of memory" dem16b_thu3.txt | tail -3
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm 3 hỏng (mã $rc) — dán dem16b_thu3.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong. Phút/lượt ở trên × 42 lượt ≈ tổng thời gian (có thể chạy sang ban ngày — cứ để chạy)."

git add -A && git commit -q -m "dem16b: dp_sau.py (DP-SGD L1/L2 trong FL) + BAN_DO dem15" || true
cat > hang_doi_dem16b.sh <<Q
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="$HAMFL_GIA_THUYET"
export HAMFL_DU_DOAN="$HAMFL_DU_DOAN"
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do for e in 3 inf 1 8 clip; do
  python dp_sau.py --backbone \$bb --muc-mo 1 --epsilon \$e --seed \$s
done; done; done
python phan_tich_dp_sau.py > dem16b_phan_tich_L1.txt 2>&1
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do for e in 3 inf; do
  python dp_sau.py --backbone \$bb --muc-mo 2 --epsilon \$e --seed \$s
done; done; done
echo "=== XONG ĐÊM 16b \$(date) ==="
python phan_tich_dp_sau.py > dem16b_phan_tich.txt 2>&1; cat dem16b_phan_tich.txt
python bao_dong.py "Đêm 16b xong" "\$(sed -n '/CỔNG A/,/Đọc/p' dem16b_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem16b.sh > dem16b.txt 2>&1 &
sleep 90; echo "--- 90 giây đầu:"; grep -v Warning dem16b.txt | tail -3
