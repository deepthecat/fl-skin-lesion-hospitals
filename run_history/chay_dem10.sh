#!/usr/bin/env bash
# Đêm 10 (17→18.10): G8b qua kiến trúc — mô hình LayerNorm (ConvNeXt-atto, DeiT-tiny): tập trung / FedAvg nguồn / FedAvg trộn × 3 seed.
set -u
cd ~/ham_fl
[ -f phan_tich_chuan_hoa.py ] || { echo "LỖI: thiếu phan_tich_chuan_hoa.py"; exit 1; }
git add -A && git commit -q -m "dem10: quet kieu chuan hoa (LN)" || true
for bb in convnext_atto deit_tiny_patch16_224; do
  HAMFL_WANDB=0 python huan_luyen_tap_trung.py --backbone $bb --tien-xu-ly p1 --do-toc 2>&1 | tail -1 || { echo "LỖI đo tốc độ $bb"; exit 1; }
  HAMFL_WANDB=0 python huan_luyen_lien_ket.py --backbone $bb --kich-ban nguon --tien-xu-ly p1 --dung-sau-vong 1 --ra /tmp/thu_d10 > /tmp/thu_d10.log 2>&1 || { tail -15 /tmp/thu_d10.log; echo "LỖI thử FL $bb"; exit 1; }
  echo "THỬ OK $bb — $(grep 'vòng   1' /tmp/thu_d10.log)"
done
rm -rf /tmp/thu_d10
cat > hang_doi_dem10.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
export HAMFL_GIA_THUYET="G8b,B2"
export HAMFL_DU_DOAN="Mô hình LayerNorm (ConvNeXt-atto, DeiT-tiny): phần khoảng cách do máy chụp nhỏ hơn ~45% của EfficientNet-B0 (BatchNorm)"
for s in 0 1 2; do
  for bb in convnext_atto deit_tiny_patch16_224; do
    python huan_luyen_tap_trung.py --backbone $bb --tien-xu-ly p1 --seed $s
    python huan_luyen_lien_ket.py  --backbone $bb --kich-ban nguon      --tien-xu-ly p1 --seed $s
    python huan_luyen_lien_ket.py  --backbone $bb --kich-ban tron_nguon --tien-xu-ly p1 --seed $s
  done
done
echo "=== XONG ĐÊM 10 $(date) ==="
python bao_dong.py "Đêm 10 xong" "$(python phan_tich_chuan_hoa.py 2>&1 | tail -6)"
Q
setsid nohup bash hang_doi_dem10.sh > dem10_chuan_hoa.txt 2>&1 &
sleep 120; echo "--- 2 phút đầu:"; grep -v Warning dem10_chuan_hoa.txt | tail -3
