exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do tap_trung --backbone $bb --cong-thuc ham --epoch 60 --workers 2 --seed $s
done; done
echo "=== XONG LƯỢT 25 $(date) ==="
python phan_tich_isic7.py > dem25_phan_tich.txt 2>&1; cat dem25_phan_tich.txt
cp dem25_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Lượt 25 xong" "$(head -6 dem25_phan_tich.txt)"
