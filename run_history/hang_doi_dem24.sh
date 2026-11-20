exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do fedavg --backbone $bb --cong-thuc ham --vong 27 --so-buoc 100 --workers 2 --seed $s
done; done
echo "=== XONG LƯỢT 24 $(date) ==="
python phan_tich_isic6.py > dem24_phan_tich.txt 2>&1; cat dem24_phan_tich.txt
cp dem24_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Lượt 24 xong" "$(head -7 dem24_phan_tich.txt)"
