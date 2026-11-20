#!/usr/bin/env bash
# Đêm 8 (15→16.10): G8 hướng cập nhật + G9 FedProx μ lớn. 12 lượt, ~3–3,5 giờ. Log: dem8_huong.txt
set -u
cd ~/ham_fl
# --- tự kiểm tra bản file trước khi bấm (bài học 12.10)
grep -q "ghi-huong" huan_luyen_lien_ket.py || { echo "LỖI: huan_luyen_lien_ket.py là bản cũ (thiếu --ghi-huong)"; exit 1; }
python -c "import pandas as pd; from du_lieu import RA; c=pd.read_csv(RA/'chia_tap.csv').columns; assert 'may_tron_nguon' in c, 'thiếu cột may_tron_nguon'" || exit 1
# --- thử 1 vòng để chắc phần ghi hướng chạy trên mô hình thật (ghi vào thư mục tạm, không đụng kết quả)
python huan_luyen_lien_ket.py --kich-ban nguon --tien-xu-ly p1 --ghi-huong --dung-sau-vong 1 --ra /tmp/thu_gh > /tmp/thu_gh.log 2>&1 || { tail -20 /tmp/thu_gh.log; echo "LỖI ở lượt thử"; exit 1; }
python -c "import torch,glob; ck=torch.load(glob.glob('/tmp/thu_gh/*_ckpt.pt')[0],weights_only=False); h=ck['lich_su'][0]; print('THỬ OK — cos_tb vòng 1:', h['cos_tb'], ' độ trôi:', h['do_troi'])" || exit 1
rm -rf /tmp/thu_gh
# --- hàng đợi chạy nền (đóng terminal vẫn chạy); flock chặn hai hàng đợi chạy song song (sự cố 01.10)
cat > hang_doi_dem8.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --kich-ban nguon      --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --kich-ban tron_nguon --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --mu 0.1 --kich-ban nguon --tien-xu-ly p1 --ghi-huong --seed $s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --mu 1   --kich-ban nguon --tien-xu-ly p1 --ghi-huong --seed $s
done
echo "=== XONG ĐÊM 8 $(date) ==="
Q
setsid nohup bash hang_doi_dem8.sh > dem8_huong.txt 2>&1 &
sleep 60
echo "--- 60 giây đầu của log:"; tail -5 dem8_huong.txt
