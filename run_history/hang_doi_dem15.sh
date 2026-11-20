exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="thiet_ke_B2,RMIA"
export HAMFL_DU_DOAN="(1) RMIA > loss: AUC tăng ≥0,03 và TPR@1%FPR tăng ≥2 lần ở cả ConvNeXt và EfficientNet. (2) ConvNeXt 50 vòng không hơn 30 vòng quá 0,02 → giữ 30 vòng"
for bb in convnext_atto efficientnet_b0; do
  for c in 0 1 2 3; do for n in 0 1; do
    python huan_luyen_tham_chieu.py --backbone $bb --cap $c --nua $n
  done; done
done
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --vong 50 --seed $s
done
echo "=== XONG HUẤN LUYỆN ĐÊM 15 $(date) ==="
python rmia.py > dem15_rmia.txt 2>&1; tail -25 dem15_rmia.txt
echo "=== XONG ĐÊM 15 $(date) ==="
python bao_dong.py "Đêm 15 xong" "$(sed -n '/Trung bình 3 seed/,$p' dem15_rmia.txt | head -12)"
