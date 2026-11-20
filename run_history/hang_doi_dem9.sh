exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
export HAMFL_GIA_THUYET="G8b,B1"
export HAMFL_DU_DOAN="FedBN nguồn thật hơn FedAvg nguồn thật (0,659) rõ rệt; trên trộn máy gần như không hơn FedAvg trộn (0,707); bn_kc nguồn thật > trộn máy; rosendahl được lợi nhiều nhất"
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --thuat-toan fedbn --kich-ban nguon      --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedbn --kich-ban tron_nguon --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedbn --kich-ban nguon      --tien-xu-ly p2 --ghi-huong --seed $s
done
echo "=== XONG ĐÊM 9 (FedBN) $(date) ==="
n=$(ls ketqua/lien_ket/fedbn_*_gh_*.json 2>/dev/null | wc -l)
python bao_dong.py "Đêm 9 xong" "FedBN: $n/9 lượt có kết quả. Xem dem9_fedbn.txt"
