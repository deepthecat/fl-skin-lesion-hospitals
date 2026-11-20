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
