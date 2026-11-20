exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
export HAMFL_GIA_THUYET="G8b,B1,B2,H4"
export HAMFL_DU_DOAN="(1) FedBN trên ResNet-18 lặp lại EfficientNet: lợi rõ ở nguồn thật, ~0 ở trộn máy. (2) ConvNeXt (LayerNorm): P2 không cải thiện FedAvg so với P1"
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --thuat-toan fedbn --backbone resnet18 --kich-ban nguon      --tien-xu-ly p1 --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedbn --backbone resnet18 --kich-ban tron_nguon --tien-xu-ly p1 --seed $s
  python huan_luyen_tap_trung.py --backbone convnext_atto --tien-xu-ly p2 --seed $s
  python huan_luyen_lien_ket.py  --backbone convnext_atto --kich-ban nguon      --tien-xu-ly p2 --seed $s
  python huan_luyen_lien_ket.py  --backbone convnext_atto --kich-ban tron_nguon --tien-xu-ly p2 --seed $s
done
echo "=== XONG ĐÊM 12 $(date) ==="
python bao_dong.py "Đêm 12 xong" "$(python phan_tich_dem12.py 2>&1 | tail -10)"
