exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
for bb in convnext_atto efficientnet_b0; do
  python dp_tuyen_tinh.py --backbone $bb --tien-xu-ly p1 --luoi --cac-kich-ban nguon
done
echo "=== XONG ĐÊM 16a $(date) ==="
python phan_tich_dp_sau.py
