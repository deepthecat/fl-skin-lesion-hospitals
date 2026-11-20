exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="G8b"
export HAMFL_DU_DOAN="Cùng ResNet-50 (ảnh 160, lô 32), chỉ đổi chuẩn hoá: BatchNorm được lợi khi trộn máy (~+0,05), GroupNorm ~0; GroupNorm ở nguồn thật cao hơn BatchNorm"
for s in 0 1 2; do
  for bb in resnet50.a1h_in1k resnet50_gn.a1h_in1k; do
    python huan_luyen_tap_trung.py --backbone $bb --bs 32 --kt 160 --tien-xu-ly p1 --seed $s
    python huan_luyen_lien_ket.py  --backbone $bb --bs 32 --kt 160 --kich-ban nguon      --tien-xu-ly p1 --seed $s
    python huan_luyen_lien_ket.py  --backbone $bb --bs 32 --kt 160 --kich-ban tron_nguon --tien-xu-ly p1 --seed $s
  done
done
echo "=== XONG ĐÊM 13 $(date) ==="
python bao_dong.py "Đêm 13 xong" "$(python phan_tich_dem13.py 2>&1 | tail -12)"
