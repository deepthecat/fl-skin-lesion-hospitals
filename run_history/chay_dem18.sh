#!/usr/bin/env bash
# Đêm 18 (chạy ngay, ~1,5–2 giờ):
#   (A) đặc trưng đóng băng nào tốt nhất cho DP lớp cuối — 6 xương sống mới × lr {4, 1} × 6 mức ε × 3 seed
#   (B) bước 4: máy chủ tò mò thấy gì, gộp bảo mật che được bao nhiêu (sim_may_chu.py)
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in dp_tuyen_tinh.py sim_may_chu.py phan_tich_dac_trung.py phan_tich_dp_sau.py rmia.py huan_luyen_lien_ket.py; do
  [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem1[0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; x=torch.ones(1000,1000,device='cuda'); print('GPU ổn:', torch.cuda.get_device_name(0))" \
  || { echo "LỖI: GPU không phản hồi. Mở PowerShell gõ: wsl --shutdown, mở lại WSL rồi chạy lại."; exit 1; }
for bb in convnext_atto efficientnet_b0; do for s in 0 1 2; do
  [ -f ketqua/lien_ket/fedavg_nguon_${bb}_txlp1_kt224_s$s.pt ] || echo "CẢNH BÁO: thiếu mô hình FedAvg $bb seed $s (.pt) — (B) sẽ bỏ qua lượt đó"
done; done

echo "=== Đêm 17 (c): L1 lô 1024 + lr 3e-4 (để ghi BAN_DO)"
python phan_tich_dp_sau.py | grep -A2 "lô1024"
F17=$(ls -t "$D"/BAN_DO_dem17*.md 2>/dev/null | head -1)
if ! grep -q "Kết quả đêm 17" BAN_DO.md && [ -n "$F17" ]; then cat "$F17" >> BAN_DO.md && echo "Đã ghi đêm 17 vào BAN_DO.md."; fi

# Trích đặc trưng theo lô 32 thay vì 128 để xương sống to (ConvNeXt-T, ResNet-50) vừa 4 GB. Đặc trưng ra y hệt.
if grep -q "range(0, len(X), 128)" dp_tuyen_tinh.py; then
  sed -i 's/range(0, len(X), 128)/range(0, len(X), 32)/; s/X\[i:i + 128\]/X[i:i + 32]/' dp_tuyen_tinh.py && echo "dp_tuyen_tinh.py: trích đặc trưng theo lô 32."
fi
python -c "import ast; ast.parse(open('dp_tuyen_tinh.py').read())" || { echo "LỖI: vá dp_tuyen_tinh.py hỏng"; exit 1; }

export HAMFL_GIA_THUYET="thiet_ke_B3,dac_trung,thiet_ke_B4"
export HAMFL_DU_DOAN="(A) xương sống học trước trên dữ liệu lớn hơn (ConvNeXt-T in22k, ViT-S in21k) cho đặc trưng đóng băng tốt hơn: hơn EfficientNet-B0 > 0,03 ở ε=3. (B) mô hình riêng của bệnh viện lộ hơn mô hình đã gộp, rõ nhất ở dias"

git add -A && git commit -q -m "dem18: dac trung dong bang cho DP L0 + sim may chu/gop bao mat; BAN_DO dem17" || true
cat > hang_doi_dem18.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for bb in resnet18 deit_tiny_patch16_224 efficientnet_b3 resnet50.a1h_in1k vit_small_patch16_224.augreg_in21k_ft_in1k convnext_tiny.fb_in22k_ft_in1k; do
  for lr in 4 1; do
    python dp_tuyen_tinh.py --backbone $bb --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr $lr --ra ketqua/dp_tuyen_tinh_lr$lr
  done
done
python phan_tich_dac_trung.py > dem18_dac_trung.txt 2>&1; cat dem18_dac_trung.txt
python sim_may_chu.py > dem18_may_chu.txt 2>&1; tail -40 dem18_may_chu.txt
echo "=== XONG ĐÊM 18 $(date) ==="
python bao_dong.py "Đêm 18 xong" "$(head -14 dem18_dac_trung.txt)"
Q
setsid nohup bash hang_doi_dem18.sh > dem18.txt 2>&1 &
sleep 90; echo "--- 90 giây đầu:"; grep -v Warning dem18.txt | tail -4
