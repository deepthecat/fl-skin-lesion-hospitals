#!/usr/bin/env bash
# Đêm 11: (a) ResNet-18 (BatchNorm thứ hai) × {tập trung, FedAvg nguồn, FedAvg trộn} × 3 seed;
#         (b) DeiT-tiny FedAvg trên 2 lần trộn khác (tron_nguon1, tron_nguon2) × 3 seed — kiểm "trộn làm tệ hơn".
set -u
cd ~/ham_fl
grep -q "KIỂM DeiT" phan_tich_chuan_hoa.py || { echo "LỖI: phan_tich_chuan_hoa.py bản cũ"; exit 1; }
python -c "import pandas as pd; from du_lieu import RA; c=pd.read_csv(RA/'chia_tap.csv').columns; assert {'may_tron_nguon1','may_tron_nguon2'}<=set(c)" || { echo "LỖI: thiếu cột trộn 1/2"; exit 1; }
git add -A && git commit -q -m "dem11: resnet18 + deit tron 1/2" || true
HAMFL_WANDB=0 python huan_luyen_tap_trung.py --backbone resnet18 --tien-xu-ly p1 --do-toc 2>&1 | tail -1 || { echo "LỖI resnet18"; exit 1; }
cat > hang_doi_dem11.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
export HAMFL_GIA_THUYET="G8b,B2"
export HAMFL_DU_DOAN="ResNet-18 (BatchNorm) giống EfficientNet: trộn máy giúp rõ (phần do máy ~30–50%); DeiT trên 2 lần trộn khác: chênh với nguồn thật trong nhiễu, tức lần trộn đầu là ngẫu nhiên"
for s in 0 1 2; do
  python huan_luyen_tap_trung.py --backbone resnet18 --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py  --backbone resnet18 --kich-ban nguon      --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py  --backbone resnet18 --kich-ban tron_nguon --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py  --backbone deit_tiny_patch16_224 --kich-ban tron_nguon1 --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py  --backbone deit_tiny_patch16_224 --kich-ban tron_nguon2 --tien-xu-ly p1 --seed $s
done
echo "=== XONG ĐÊM 11 $(date) ==="
python bao_dong.py "Đêm 11 xong" "$(python phan_tich_chuan_hoa.py 2>&1 | tail -8)"
Q
setsid nohup bash hang_doi_dem11.sh > dem11.txt 2>&1 &
sleep 120; echo "--- 2 phút đầu:"; grep -v Warning dem11.txt | tail -3
