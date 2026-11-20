#!/usr/bin/env bash
# Đêm 13: ResNet-50 BatchNorm vs ResNet-50 GroupNorm — cùng kiến trúc, cùng công thức học trước a1h, chỉ đổi chuẩn hoá.
# {tập trung, FedAvg nguồn, FedAvg trộn máy} × 2 mô hình × 3 seed = 18 lượt.
# Vì RTX 3050 4 GB: ảnh 160 (thay 224), lô 32 (thay 64) cho CẢ HAI mô hình → so sánh TRONG CẶP vẫn công bằng,
# nhưng không so trực tiếp số tuyệt đối với các mô hình chạy ở 224.
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
KT=160; BS=32
BN=resnet50.a1h_in1k; GN=resnet50_gn.a1h_in1k
[ -f phan_tich_dem13.py ] || { echo "LỖI: thiếu phan_tich_dem13.py"; exit 1; }
[ -f hang_doi_dem13.sh ] && pgrep -f hang_doi_dem13.sh >/dev/null && { echo "Đêm 13 đang chạy rồi — thoát."; exit 1; }

# ---------- 0. Ghi chép BAN_DO.md (mỗi phần một lần; trình duyệt có thể đổi tên thành "... (1).md" nên lấy bản mới nhất)
moi_nhat() { ls -t "$D"/$1*.md 2>/dev/null | head -1; }
F11=$(moi_nhat BAN_DO_phan_them_sau_16-10)
if ! grep -q "Kết quả đêm 11" BAN_DO.md; then
  if [ -n "$F11" ] && grep -q "Kết quả đêm 11" "$F11"; then
    # chèn TRƯỚC phần đêm 12 nếu đêm 12 đã được ghi, để giữ đúng thứ tự thời gian
    python - "$F11" <<'PY'
import sys, pathlib
p = pathlib.Path("BAN_DO.md"); cu = p.read_text(encoding="utf-8")
moi = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
moc = "\n### Kết quả đêm 12"
p.write_text(cu.replace(moc, "\n" + moi.rstrip() + "\n" + moc, 1) if moc in cu else cu + moi, encoding="utf-8")
PY
    echo "Đã ghi phần sau 16.10 (đến đêm 11) từ: $(basename "$F11")"
  else
    echo "CẢNH BÁO: chưa có BAN_DO_phan_them_sau_16-10*.md trong Research_2026_Code (không ảnh hưởng việc chạy)."
  fi
fi
F12=$(moi_nhat BAN_DO_dem12)
if ! grep -q "Kết quả đêm 12" BAN_DO.md && [ -n "$F12" ]; then cat "$F12" >> BAN_DO.md && echo "Đã ghi đêm 12."; fi

export HAMFL_GIA_THUYET="G8b"
export HAMFL_DU_DOAN="Cùng ResNet-50 (ảnh 160, lô 32), chỉ đổi chuẩn hoá: BatchNorm được lợi khi trộn máy (~+0,05), GroupNorm ~0; GroupNorm ở nguồn thật cao hơn BatchNorm"

# ---------- 1. Tự kiểm VRAM + tốc độ (tải trọng số học trước nếu chưa có). Im lặng 2–5 phút.
echo ">>> Tự kiểm 1/2: tải trọng số + đo VRAM ở ảnh $KT, lô $BS. Im lặng 2–5 phút, đừng Ctrl+C."
TONG=0
for bb in $GN $BN; do
  out=$(python huan_luyen_tap_trung.py --backbone $bb --bs $BS --kt $KT --tien-xu-ly p1 --do-toc 2>&1 | grep -v Warning | tail -1)
  echo "    $out"
  vram=$(echo "$out" | grep -oP 'VRAM đỉnh \K[0-9.]+'); sep=$(echo "$out" | grep -oP '~\K[0-9]+(?= s/epoch)')
  [ -n "$vram" ] && [ -n "$sep" ] || { echo "LỖI: $bb không chạy được khi đo — dán output cho Claude"; exit 1; }
  awk "BEGIN{exit !($vram < 3.4)}" || { echo "LỖI: $bb cần $vram GB, quá sát 4 GB — dán output cho Claude"; exit 1; }
  TONG=$((TONG + sep))
done
# mỗi (mô hình, seed): 15 epoch tập trung + 30 vòng × 2 kịch bản FL ≈ 75 epoch, cộng ~15% cho đánh giá
GIO=$(awk "BEGIN{printf \"%.1f\", 3*75*$TONG*1.15/3600}")
echo ">>> Ước tính toàn bộ đêm 13: ~$GIO giờ."

# ---------- 2. Chạy thử FL thật 1 vòng (kiểm đường dữ liệu ảnh 160, tên mô hình, checkpoint, wandb).
#      Lượt thật sau đó sẽ CHẠY TIẾP từ checkpoint vòng 1 này — không mất công.
echo ">>> Tự kiểm 2/2: chạy thử 1 vòng FL với GroupNorm (~3–5 phút)."
python huan_luyen_lien_ket.py --backbone $GN --bs $BS --kt $KT --kich-ban nguon --tien-xu-ly p1 --seed 0 --dung-sau-vong 1 > dem13_thu.txt 2>&1
rc=$?
grep -v Warning dem13_thu.txt | tail -4
[ $rc -eq 0 ] || { echo "LỖI: chạy thử FL hỏng (mã $rc) — dán dem13_thu.txt cho Claude"; exit 1; }
ls ketqua/lien_ket/ | grep -q "fedavg_nguon_${GN}_txlp1_kt${KT}_s0_ckpt.pt" \
  || { echo "LỖI: không thấy checkpoint sau khi chạy thử — dán dem13_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong, mọi thứ ổn."

git add -A && git commit -q -m "dem13: resnet50 BN vs GN (kt$KT bs$BS)" || true
cat > hang_doi_dem13.sh <<Q
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="$HAMFL_GIA_THUYET"
export HAMFL_DU_DOAN="$HAMFL_DU_DOAN"
for s in 0 1 2; do
  for bb in $BN $GN; do
    python huan_luyen_tap_trung.py --backbone \$bb --bs $BS --kt $KT --tien-xu-ly p1 --seed \$s
    python huan_luyen_lien_ket.py  --backbone \$bb --bs $BS --kt $KT --kich-ban nguon      --tien-xu-ly p1 --seed \$s
    python huan_luyen_lien_ket.py  --backbone \$bb --bs $BS --kt $KT --kich-ban tron_nguon --tien-xu-ly p1 --seed \$s
  done
done
echo "=== XONG ĐÊM 13 \$(date) ==="
python bao_dong.py "Đêm 13 xong" "\$(python phan_tich_dem13.py 2>&1 | tail -12)"
Q
setsid nohup bash hang_doi_dem13.sh > dem13.txt 2>&1 &
sleep 120; echo "--- 2 phút đầu:"; grep -v Warning dem13.txt | tail -3
