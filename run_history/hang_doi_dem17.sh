exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="thiet_ke_B3,cong_A"
export HAMFL_DU_DOAN="(b) lr nhỏ hơn cải thiện L0 có DP ít nhất 0,03 ở ε=3. (c) L1 lô 1024 vẫn kém L0 ở ε=3 nhưng không còn sụp về đoán mò ở ε=8"
# (a) chạy lại lượt hỏng — lượt đã có json tự bỏ qua
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  for e in 3 inf 1 8 clip; do python dp_sau.py --backbone $bb --muc-mo 1 --epsilon $e --seed $s; done
  for e in 3 inf; do python dp_sau.py --backbone $bb --muc-mo 2 --epsilon $e --seed $s; done
done; done
# (b) L0 với lr nhỏ hơn
for lr in 4 1; do for bb in convnext_atto efficientnet_b0; do
  python dp_tuyen_tinh.py --backbone $bb --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr $lr --ra ketqua/dp_tuyen_tinh_lr$lr
done; done
python phan_tich_dp_sau.py > dem17_phan_tich_ab.txt 2>&1
# (c) L1 lô 1024, lr 3e-4
for e in 8 3 inf; do for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python dp_sau.py --backbone $bb --muc-mo 1 --epsilon $e --bs 1024 --lr 3e-4 --seed $s
done; done; done
echo "=== XONG ĐÊM 17 $(date) ==="
python phan_tich_dp_sau.py > dem17_phan_tich.txt 2>&1; cat dem17_phan_tich.txt
python bao_dong.py "Đêm 17 xong" "$(sed -n '/CỔNG A/,/Đọc/p' dem17_phan_tich.txt)"
