#!/usr/bin/env bash
# GĐ1: benchmark Fed-ISIC2019 theo giao thức FLamby. 12 lượt (4 chế độ × 3 seed). Log: isic1.txt
set -u
cd ~/ham_fl
grep -q "FocalLoss" fed_isic.py || { echo "LỖI: thiếu fed_isic.py bản mới"; exit 1; }
[ -f ketqua/fed_isic_chia.csv ] || { echo "LỖI: chưa chạy chuan_bi_isic.py"; exit 1; }
pip install -q "albumentations==1.3.1" efficientnet_pytorch || { echo "LỖI cài thư viện"; exit 1; }
git add -A && git commit -q -m "fed_isic: benchmark Fed-ISIC2019 (FLamby)" || true
# --- thử nhanh: 20 ảnh/trung tâm, 1 epoch, CÓ tải trọng số học trước (kiểm mạng), thư mục tạm
HAMFL_WANDB=0 python fed_isic.py --che-do tap_trung --epoch 1 --gioi-han-anh 20 --ra /tmp/thu_isic > /tmp/thu_isic.log 2>&1 \
  && HAMFL_WANDB=0 python fed_isic.py --che-do fedbn --vong 1 --so-buoc 3 --gioi-han-anh 20 --ra /tmp/thu_isic >> /tmp/thu_isic.log 2>&1 \
  || { tail -25 /tmp/thu_isic.log; echo "LỖI ở lượt thử"; exit 1; }
grep -c "KẾT QUẢ" /tmp/thu_isic.log | xargs -I{} echo "THỬ OK — {}/2 lượt thử chạy hết"; rm -rf /tmp/thu_isic
cat > hang_doi_isic1.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
export HAMFL_GIA_THUYET="D1"
export HAMFL_DU_DOAN="Tái tạo FLamby: tập trung ~0,68, FedAvg ~0,64 (số công bố 0,683 / 0,644); khoảng ~0,04, nhỏ hơn nhiều so với HAM10000 4 nguồn (0,119) vì có BCN rất lớn; FedBN chấm theo trung tâm > FedAvg"
for s in 0 1 2; do
  for m in tap_trung fedavg fedbn fedprox; do python fed_isic.py --che-do $m --seed $s; done
done
echo "=== XONG ISIC1 $(date) ==="
python bao_dong.py "Fed-ISIC2019 xong" "$(ls ketqua/fed_isic/*.json 2>/dev/null | wc -l)/12 lượt. Xem isic1.txt"
Q
setsid nohup bash hang_doi_isic1.sh > isic1.txt 2>&1 &
sleep 240
echo "--- 4 phút đầu:"; grep -v Warning isic1.txt | tail -4
