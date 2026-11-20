exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
export HAMFL_GIA_THUYET="D1"
export HAMFL_DU_DOAN="Tái tạo FLamby: tập trung ~0,68, FedAvg ~0,64 (số công bố 0,683 / 0,644); khoảng ~0,04, nhỏ hơn nhiều so với HAM10000 4 nguồn (0,119) vì có BCN rất lớn; FedBN chấm theo trung tâm > FedAvg"
for s in 0 1 2; do
  for m in tap_trung fedavg fedbn fedprox; do python fed_isic.py --che-do $m --seed $s; done
done
echo "=== XONG ISIC1 $(date) ==="
python bao_dong.py "Fed-ISIC2019 xong" "$(ls ketqua/fed_isic/*.json 2>/dev/null | wc -l)/12 lượt. Xem isic1.txt"
