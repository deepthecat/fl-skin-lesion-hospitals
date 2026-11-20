exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="thiet_ke_B3,cong_A"
export HAMFL_DU_DOAN="Cổng A: ở ε=3 và ε=1, ConvNeXt (L1 hoặc L2) có val bAcc cao hơn EfficientNet L0; DP sâu hơn (L1) tốt hơn L0 khi ε≥3, kém hơn khi ε=1"
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do for e in 3 inf 1 8 clip; do
  python dp_sau.py --backbone $bb --muc-mo 1 --epsilon $e --seed $s
done; done; done
python phan_tich_dp_sau.py > dem16b_phan_tich_L1.txt 2>&1
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do for e in 3 inf; do
  python dp_sau.py --backbone $bb --muc-mo 2 --epsilon $e --seed $s
done; done; done
echo "=== XONG ĐÊM 16b $(date) ==="
python phan_tich_dp_sau.py > dem16b_phan_tich.txt 2>&1; cat dem16b_phan_tich.txt
python bao_dong.py "Đêm 16b xong" "$(sed -n '/CỔNG A/,/Đọc/p' dem16b_phan_tich.txt)"
