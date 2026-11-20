exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do tap_trung --backbone $bb --cong-thuc ham --workers 2 --seed $s
  python fed_isic.py --che-do fedavg    --backbone $bb --cong-thuc ham --workers 2 --seed $s
done; done
echo "=== XONG ĐÊM 22 $(date) ==="
python phan_tich_isic4.py > dem22_phan_tich.txt 2>&1; cat dem22_phan_tich.txt
python bao_dong.py "Đêm 22 xong" "$(head -8 dem22_phan_tich.txt)"
