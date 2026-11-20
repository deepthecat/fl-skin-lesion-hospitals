#!/usr/bin/env bash
# Đêm 21 (~3 giờ): (1) trộn máy trên Fed-ISIC2019 — phép thử G8b trên bộ dữ liệu thứ hai, không lẫn yếu tố kiến trúc;
#                  (2) seed 3, 4 cho các con số chính trên HAM10000 (5 seed + khoảng tin cậy 95%).
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in fed_isic.py va_isic.py va_isic2.py tao_tron_isic.py phan_tich_isic3.py phan_tich_5seed.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
python va_isic.py && python va_isic2.py || exit 1
python -c "import pandas as pd; from du_lieu import RA; import sys; sys.exit(0 if 'may_tron' in pd.read_csv(RA/'fed_isic_chia.csv').columns else 1)" \
  || python tao_tron_isic.py | head -9
F20=$(ls -t "$D"/BAN_DO_dem20*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 20" BAN_DO.md && [ -n "$F20" ]; then cat "$F20" >> BAN_DO.md && echo "Đã ghi đêm 20 vào BAN_DO.md."; fi
export HAMFL_GIA_THUYET="G8b,du_lieu_thu_hai,5_seed"
export HAMFL_DU_DOAN="Fed-ISIC: trộn máy làm FedAvg tăng > 0,03 ở cả 3 seed (MSK, rosendahl tăng nhiều nhất). 5 seed không đổi kết luận nào của 3 seed"
echo ">>> Tự kiểm: FedAvg trộn máy, 1 vòng × 5 bước, 50 ảnh/trung tâm (thư mục thử)."
python fed_isic.py --che-do fedavg --kich-ban tron --vong 1 --so-buoc 5 --gioi-han-anh 50 --workers 2 --seed 0 --ra ketqua/fed_isic_thu > dem21_thu.txt 2>&1
rc=$?; grep -E "trung tâm|KẾT QUẢ|Error|Traceback" dem21_thu.txt | tail -3
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem21_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong."
git add -A && git commit -q -m "dem21: tron may Fed-ISIC + 5 seed HAM" || true
cat > hang_doi_dem21.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do python fed_isic.py --che-do fedavg --kich-ban tron --workers 2 --seed $s; done
python phan_tich_isic3.py > dem21_isic.txt 2>&1
for s in 3 4; do
  for bb in convnext_atto efficientnet_b0; do
    python huan_luyen_tap_trung.py --backbone $bb --tien-xu-ly p1 --seed $s
    python huan_luyen_lien_ket.py --backbone $bb --kich-ban nguon --tien-xu-ly p1 --seed $s
  done
  for e in 8 3 1; do
    python dp_tuyen_tinh.py --backbone efficientnet_b0 --tien-xu-ly p1 --kich-ban nguon --epsilon $e --lr 4 --seed $s --ra ketqua/dp_tuyen_tinh_lr4
  done
done
echo "=== XONG ĐÊM 21 $(date) ==="
cat dem21_isic.txt; python phan_tich_5seed.py > dem21_5seed.txt 2>&1; cat dem21_5seed.txt
python bao_dong.py "Đêm 21 xong" "$(head -6 dem21_isic.txt)"
Q
setsid nohup bash hang_doi_dem21.sh > dem21.txt 2>&1 &
sleep 90; echo "--- 90 giây đầu:"; grep -v Warning dem21.txt | tail -3; free -h | head -2
