#!/usr/bin/env bash
# Đêm 12 (chạy ban ngày 19.10): (1) FedBN trên ResNet-18 — nguồn thật & trộn máy × 3 seed;
# (2) ConvNeXt-atto với cân màu P2 — tập trung, FedAvg nguồn, FedAvg trộn × 3 seed. 15 lượt, ~3–4 giờ.
set -u
cd ~/ham_fl
grep -q '"fedbn"' huan_luyen_lien_ket.py || { echo "LỖI: huan_luyen_lien_ket.py thiếu fedbn"; exit 1; }
[ -f phan_tich_dem12.py ] || { echo "LỖI: thiếu phan_tich_dem12.py"; exit 1; }
git add -A && git commit -q -m "dem12: fedbn resnet18 + convnext P2" || true
cat > hang_doi_dem12.sh <<'Q'
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
Q
setsid nohup bash hang_doi_dem12.sh > dem12.txt 2>&1 &
sleep 90; echo "--- 90 giây đầu:"; grep -v Warning dem12.txt | tail -3
