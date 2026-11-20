exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="G9,thiet_ke_B1"
export HAMFL_DU_DOAN="ConvNeXt (LayerNorm): FedProx ≈ FedAvg (trôi không phải nút nghẽn, như EfficientNet); FedAvgM ≈ FedAvg hoặc kém hơn. Không thuật toán nào đạt luật chọn → giữ FedAvg"
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --thuat-toan fedavgm --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --mu 0.1 --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --seed $s
done
echo "=== XONG ĐÊM 14 $(date) ==="
python bao_dong.py "Đêm 14 xong" "$(python phan_tich_dem14.py 2>&1 | sed -n '3,14p')"
