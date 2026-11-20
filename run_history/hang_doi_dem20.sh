exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do fedavg --backbone $bb --toi-uu-fl adam --seed $s --workers 2
  python fed_isic.py --che-do tap_trung --backbone $bb --seed $s --workers 2
done; done
echo "=== XONG ĐÊM 20 $(date) ==="
python phan_tich_isic2.py > dem20_phan_tich.txt 2>&1; cat dem20_phan_tich.txt
python bao_dong.py "Đêm 20 xong" "$(head -8 dem20_phan_tich.txt)"
