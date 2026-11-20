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
