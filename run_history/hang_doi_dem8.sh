exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --kich-ban nguon      --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --kich-ban tron_nguon --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --mu 0.1 --kich-ban nguon --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --mu 1   --kich-ban nguon --tien-xu-ly p1 --ghi-huong --seed $s
done
echo "=== XONG ĐÊM 8 $(date) ==="
