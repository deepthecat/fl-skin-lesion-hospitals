exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in efficientnet_b0 convnext_atto; do
  python fed_isic.py --che-do fedavg --backbone $bb --cong-thuc ham --kich-ban tron --vong 27 --so-buoc 100 --workers 2 --seed $s
done; done
echo "=== XONG LƯỢT 26 $(date) ==="
python phan_tich_isic8.py > dem26_phan_tich.txt 2>&1; cat dem26_phan_tich.txt
cp dem26_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Lượt 26 xong" "$(head -6 dem26_phan_tich.txt)"
