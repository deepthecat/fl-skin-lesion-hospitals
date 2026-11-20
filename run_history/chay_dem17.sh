#!/usr/bin/env bash
# Đêm 17 — kiểm "DP sâu sụp là do công thức hay do bản chất" trước khi chốt cổng A.
#   (a) chạy lại 2 lượt đêm 16b hỏng vì lỗi CUDA (các lượt đã có tự bỏ qua)
#   (b) L0 (dp_tuyen_tinh) với lr cố định 1 và 4 (thay vì 16 chọn khi KHÔNG DP) — 2 mô hình, nhanh
#   (c) L1 với lô kỳ vọng 1024 + lr 3e-4, ε = 8, 3, rồi ∞ — 2 mô hình × 3 seed (~3–4 giờ)
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in dp_sau.py dp_tuyen_tinh.py phan_tich_dp_sau.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
grep -q '_b{arg.bs}' dp_sau.py || { echo "LỖI: dp_sau.py chưa phải bản mới (thiếu hậu tố lô/lr) — chép lại từ Research_2026_Code"; exit 1; }
pgrep -f "hang_doi_dem1[0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; x=torch.ones(1000,1000,device='cuda'); print('GPU ổn:', torch.cuda.get_device_name(0), float((x@x).sum()))" \
  || { echo "LỖI: GPU không phản hồi (sau lỗi CUDA đêm qua). Mở PowerShell gõ: wsl --shutdown, rồi mở lại WSL và chạy lại lệnh này."; exit 1; }
F16=$(ls -t "$D"/BAN_DO_dem16*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 16" BAN_DO.md && [ -n "$F16" ]; then cat "$F16" >> BAN_DO.md && echo "Đã ghi đêm 16 vào BAN_DO.md."; fi

export HAMFL_GIA_THUYET="thiet_ke_B3,cong_A"
export HAMFL_DU_DOAN="(b) lr nhỏ hơn cải thiện L0 có DP ít nhất 0,03 ở ε=3. (c) L1 lô 1024 vẫn kém L0 ở ε=3 nhưng không còn sụp về đoán mò ở ε=8"

echo ">>> Tự kiểm: 1 vòng L1 lô 1024 (~1 phút)."
python dp_sau.py --backbone convnext_atto --muc-mo 1 --epsilon 8 --bs 1024 --lr 3e-4 --seed 0 --dung-sau-vong 1 > dem17_thu.txt 2>&1
rc=$?; grep -E "bệnh viện|dừng thử|Error" dem17_thu.txt | tail -6
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem17_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong."

git add -A && git commit -q -m "dem17: dp_sau hau to lo/lr, L0 lr nho, L1 lo 1024; BAN_DO dem16" || true
cat > hang_doi_dem17.sh <<Q
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="$HAMFL_GIA_THUYET"
export HAMFL_DU_DOAN="$HAMFL_DU_DOAN"
# (a) chạy lại lượt hỏng — lượt đã có json tự bỏ qua
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  for e in 3 inf 1 8 clip; do python dp_sau.py --backbone \$bb --muc-mo 1 --epsilon \$e --seed \$s; done
  for e in 3 inf; do python dp_sau.py --backbone \$bb --muc-mo 2 --epsilon \$e --seed \$s; done
done; done
# (b) L0 với lr nhỏ hơn
for lr in 4 1; do for bb in convnext_atto efficientnet_b0; do
  python dp_tuyen_tinh.py --backbone \$bb --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr \$lr --ra ketqua/dp_tuyen_tinh_lr\$lr
done; done
python phan_tich_dp_sau.py > dem17_phan_tich_ab.txt 2>&1
# (c) L1 lô 1024, lr 3e-4
for e in 8 3 inf; do for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python dp_sau.py --backbone \$bb --muc-mo 1 --epsilon \$e --bs 1024 --lr 3e-4 --seed \$s
done; done; done
echo "=== XONG ĐÊM 17 \$(date) ==="
python phan_tich_dp_sau.py > dem17_phan_tich.txt 2>&1; cat dem17_phan_tich.txt
python bao_dong.py "Đêm 17 xong" "\$(sed -n '/CỔNG A/,/Đọc/p' dem17_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem17.sh > dem17.txt 2>&1 &
sleep 60; echo "--- 60 giây đầu:"; grep -v Warning dem17.txt | tail -3
