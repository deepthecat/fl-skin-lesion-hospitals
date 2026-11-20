exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
# Thiết kế 2 trước (nhanh, vài chục phút): EfficientNet-B0 L0 + DP
e=efficientnet_b0
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr 4 --vong 10 --ra ketqua/dp_tuyen_tinh_lr4
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr 4 --vong 60 --ra ketqua/dp_tuyen_tinh_lr4
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p2 --luoi --cac-kich-ban nguon --lr 4 --ra ketqua/dp_tuyen_tinh_lr4
python dp_tuyen_tinh.py --backbone $e --tien-xu-ly p1 --luoi --cac-kich-ban mot_may --lr 4 --ra ketqua/dp_tuyen_tinh_lr4
python phan_tich_ablation.py > dem19_ablation_TK2.txt 2>&1
# Thiết kế 1: ConvNeXt học toàn bộ, FedAvg nguồn thật
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --khong-pretrained --seed $s
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --trong-so khong --seed $s
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --trong-so-cuc-bo --seed $s
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --epoch-cuc-bo 2 --seed $s
done
echo "=== XONG ĐÊM 19 $(date) ==="
python phan_tich_ablation.py > dem19_ablation.txt 2>&1; cat dem19_ablation.txt
python bao_dong.py "Đêm 19 xong" "$(grep -E 'GỐC|CÓ lý do|TỐT HƠN' dem19_ablation.txt | head -15)"
