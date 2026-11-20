#!/usr/bin/env bash
# Đêm 22 (~5–7 giờ): ConvNeXt-atto vs EfficientNet-B0 trên Fed-ISIC2019 với CÙNG công thức huấn luyện HAM10000.
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in fed_isic.py va_isic.py va_isic2.py va_isic3.py phan_tich_isic4.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
python va_isic.py && python va_isic2.py && python va_isic3.py || exit 1
F21=$(ls -t "$D"/BAN_DO_dem21*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 21" BAN_DO.md && [ -n "$F21" ]; then cat "$F21" >> BAN_DO.md && echo "Đã ghi đêm 21 vào BAN_DO.md."; fi
export HAMFL_GIA_THUYET="G8b,thiet_ke_1,du_lieu_thu_hai"
export HAMFL_DU_DOAN="Cùng công thức HAM: ConvNeXt tập trung > 0,70; khoảng tập trung − FedAvg của ConvNeXt nhỏ hơn EfficientNet"
echo ">>> Tự kiểm: ConvNeXt tập trung công thức HAM, 1 epoch, 50 ảnh/trung tâm (thư mục thử)."
python fed_isic.py --che-do tap_trung --backbone convnext_atto --cong-thuc ham --epoch 1 --gioi-han-anh 50 --workers 2 --seed 0 --ra ketqua/fed_isic_thu > dem22_thu.txt 2>&1
rc=$?; grep -E "epoch|KẾT QUẢ|Error|Traceback" dem22_thu.txt | tail -3
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem22_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong."
git add -A && git commit -q -m "dem22: Fed-ISIC cong thuc HAM, ConvNeXt vs EfficientNet" || true
cat > hang_doi_dem22.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do tap_trung --backbone $bb --cong-thuc ham --workers 2 --seed $s
  python fed_isic.py --che-do fedavg    --backbone $bb --cong-thuc ham --workers 2 --seed $s
done; done
echo "=== XONG ĐÊM 22 $(date) ==="
python phan_tich_isic4.py > dem22_phan_tich.txt 2>&1; cat dem22_phan_tich.txt
python bao_dong.py "Đêm 22 xong" "$(head -8 dem22_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem22.sh > dem22.txt 2>&1 &
sleep 120; echo "--- 2 phút đầu:"; grep -v Warning dem22.txt | tail -3; free -h | head -2
