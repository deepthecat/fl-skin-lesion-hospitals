exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do fedavg --backbone $bb --cong-thuc ham --vong 27 --so-buoc 33 --workers 2 --seed $s
done; done
echo "=== XONG ĐÊM 23 $(date) ==="
python phan_tich_isic5.py > dem23_phan_tich.txt 2>&1; cat dem23_phan_tich.txt
python bao_dong.py "Đêm 23 xong" "$(head -6 dem23_phan_tich.txt)"
