#!/usr/bin/env bash
# Đêm 20: Thiết kế 1 trên bộ dữ liệu THỨ HAI (Fed-ISIC2019). ConvNeXt-atto vs EfficientNet-B0 (timm),
# tập trung + FedAvg (Adam 5e-4 tại chỗ), 3 seed = 12 lượt (~5–7 giờ, chạy qua đêm được).
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in fed_isic.py va_isic.py phan_tich_isic2.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
[ -f ketqua/fed_isic_chia.csv ] || { echo "LỖI: thiếu ketqua/fed_isic_chia.csv"; exit 1; }
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
python va_isic.py || exit 1
F19=$(ls -t "$D"/BAN_DO_dem19*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 19" BAN_DO.md && [ -n "$F19" ]; then cat "$F19" >> BAN_DO.md && echo "Đã ghi đêm 19 vào BAN_DO.md."; fi
export HAMFL_GIA_THUYET="G8b,thiet_ke_1,du_lieu_thu_hai"
export HAMFL_DU_DOAN="Fed-ISIC2019: khoảng tập trung − FedAvg của ConvNeXt-atto nhỏ hơn của EfficientNet-B0 (cùng timm, cùng Adam)"
echo ">>> Tự kiểm: ConvNeXt FedAvg Adam, 1 vòng × 5 bước, 50 ảnh/trung tâm (thư mục thử)."
python fed_isic.py --che-do fedavg --backbone convnext_atto --toi-uu-fl adam --vong 1 --so-buoc 5 --gioi-han-anh 50 --seed 0 --ra ketqua/fed_isic_thu > dem20_thu.txt 2>&1
rc=$?; grep -E "vòng|KẾT QUẢ|Error|Traceback" dem20_thu.txt | tail -3
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem20_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong."
git add -A && git commit -q -m "dem20: Fed-ISIC2019 ConvNeXt vs EfficientNet (timm, Adam)" || true
cat > hang_doi_dem20.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do fedavg --backbone $bb --toi-uu-fl adam --seed $s
  python fed_isic.py --che-do tap_trung --backbone $bb --seed $s
done; done
echo "=== XONG ĐÊM 20 $(date) ==="
python phan_tich_isic2.py > dem20_phan_tich.txt 2>&1; cat dem20_phan_tich.txt
python bao_dong.py "Đêm 20 xong" "$(head -8 dem20_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem20.sh > dem20.txt 2>&1 &
sleep 120; echo "--- 2 phút đầu:"; grep -v Warning dem20.txt | tail -3
