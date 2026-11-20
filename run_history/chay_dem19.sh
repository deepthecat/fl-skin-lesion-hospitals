#!/usr/bin/env bash
# Đêm 19 — CỔNG B: rút từng thành phần khỏi hai thiết kế cuối (~3 giờ).
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in va_ten.py phan_tich_ablation.py dp_tuyen_tinh.py huan_luyen_lien_ket.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem1[0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
python va_ten.py || exit 1
F18=$(ls -t "$D"/BAN_DO_dem18*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 17c" BAN_DO.md && [ -n "$F18" ]; then cat "$F18" >> BAN_DO.md && echo "Đã ghi đêm 17c/18/18b vào BAN_DO.md."; fi
export HAMFL_GIA_THUYET="cong_B"
export HAMFL_DU_DOAN="TK1: bỏ học trước tụt mạnh (>0,1); bỏ trọng số lớp tụt >0,02; trọng số tại chỗ và 2 epoch trong nhiễu. TK2: 10 vòng tốt hơn 30 vòng ở ε=3 (ít bước -> ít nhiễu); P2 không đổi"
echo ">>> Tự kiểm: 1 vòng ConvNeXt không học trước (~1 phút)."
python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --khong-pretrained --seed 0 --dung-sau-vong 1 > dem19_thu.txt 2>&1
rc=$?; grep -E "vòng|dừng|Error" dem19_thu.txt | tail -2
[ $rc -eq 0 ] && ls ketqua/lien_ket/ | grep -q "fedavg_nguon_convnext_atto_txlp1_khp_kt224_s0_ckpt.pt" \
  || { echo "LỖI: tự kiểm hỏng — dán dem19_thu.txt cho Claude"; exit 1; }
git add -A && git commit -q -m "dem19: cong B ablation + va ten run" || true
cat > hang_doi_dem19.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
# Thiết kế 2 trước (nhanh, vài chục phút): EfficientNet-B0 L0 + DP
e=efficientnet_b0
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr 4 --vong 10 --ra ketqua/dp_tuyen_tinh_lr4
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr 4 --vong 60 --ra ketqua/dp_tuyen_tinh_lr4
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p2 --luoi --cac-kich-ban nguon --lr 4 --ra ketqua/dp_tuyen_tinh_lr4
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p1 --luoi --cac-kich-ban mot_may --lr 4 --ra ketqua/dp_tuyen_tinh_lr4
python phan_tich_ablation.py > dem19_ablation_TK2.txt 2>&1
# Thiết kế 1: ConvNeXt học toàn bộ, FedAvg nguồn thật
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --khong-pretrained --seed $s
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --trong-so khong --seed $s
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --trong-so-cuc-bo --seed $s
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --epoch-cuc-bo 2 --seed $s
done
echo "=== XONG ĐÊM 19 $(date) ==="
python phan_tich_ablation.py > dem19_ablation.txt 2>&1; cat dem19_ablation.txt
python bao_dong.py "Đêm 19 xong" "$(grep -E 'GỐC|CÓ lý do|TỐT HƠN' dem19_ablation.txt | head -15)"
Q
setsid nohup bash hang_doi_dem19.sh > dem19.txt 2>&1 &
sleep 60; echo "--- 60 giây đầu:"; grep -v Warning dem19.txt | tail -3
