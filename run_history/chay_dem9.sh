#!/usr/bin/env bash
# Đêm 9 (16→17.10): (GPU) FedBN — kiểm G8b; (mạng, song song) tải ISIC 2019 cho Fed-ISIC2019.
# Log: dem9_fedbn.txt, dem9_tai_isic.txt
set -u
cd ~/ham_fl
# --- tự kiểm tra bản file (bài học 12.10)
grep -q '"fedbn"' huan_luyen_lien_ket.py || { echo "LỖI: huan_luyen_lien_ket.py chưa có fedbn"; exit 1; }
grep -q "NhatKy" huan_luyen_lien_ket.py || { echo "LỖI: thiếu nhật ký wandb"; exit 1; }
[ -f bao_dong.py ] || { echo "LỖI: thiếu bao_dong.py"; exit 1; }
python -c "import pandas as pd; from du_lieu import RA; c=pd.read_csv(RA/'chia_tap.csv').columns; assert {'may_nguon','may_tron_nguon','nguon'}<=set(c)" || { echo "LỖI: thiếu cột trong chia_tap.csv"; exit 1; }
git add -A && git commit -q -m "dem9: fedbn + tai isic2019" || true
# --- thử 1 vòng FedBN thật (thư mục tạm, không ghi wandb)
HAMFL_WANDB=0 python huan_luyen_lien_ket.py --thuat-toan fedbn --kich-ban nguon --tien-xu-ly p1 --ghi-huong --dung-sau-vong 1 --ra /tmp/thu_fedbn > /tmp/thu_fedbn.log 2>&1 || { tail -20 /tmp/thu_fedbn.log; echo "LỖI ở lượt thử FedBN"; exit 1; }
python -c "import torch,glob; ck=torch.load(glob.glob('/tmp/thu_fedbn/*_ckpt.pt')[0],weights_only=False); h=ck['lich_su'][0]; print('THỬ OK — val theo nguồn', round(h['val_bacc'],3), '| ensemble', round(h['val_ens_bacc'],3), '| bn_kc_tb', h['bn_kc_tb'])" || exit 1
rm -rf /tmp/thu_fedbn

# --- hàng đợi GPU
cat > hang_doi_dem9.sh <<'Q'
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
Q
setsid nohup bash hang_doi_dem9.sh > dem9_fedbn.txt 2>&1 &

# --- tải ISIC 2019 (song song, chỉ dùng mạng + đĩa; KHÔNG giải nén/đổi cỡ — làm sau khi đọc resize_images.py)
cat > tai_isic.sh <<'Q'
cd ~/ham_fl
F=~/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py
DICH=~/du_lieu_ngoai/isic2019_raw; mkdir -p $DICH
python - "$F" > /tmp/isic_urls.txt <<'P'
import ast, sys
cay = ast.parse(open(sys.argv[1]).read())
for n in cay.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").startswith("url_"):
        print(ast.literal_eval(n.value))
P
echo "Sẽ tải:"; cat /tmp/isic_urls.txt
while read -r u; do wget -c -q --show-progress -P $DICH "$u" 2>&1 | tail -1; done < /tmp/isic_urls.txt
ls -la $DICH; echo "=== XONG TẢI ISIC 2019 $(date) ==="
python bao_dong.py "Tải ISIC 2019 xong" "$(ls $DICH | tr '\n' ' ')"
Q
setsid nohup bash tai_isic.sh > dem9_tai_isic.txt 2>&1 &
sleep 90
echo "--- FedBN, 90 giây đầu:"; tail -3 dem9_fedbn.txt
echo "--- Tải ISIC:"; head -5 dem9_tai_isic.txt
