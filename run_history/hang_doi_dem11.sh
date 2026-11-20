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
