#!/usr/bin/env bash
# Đêm 15 — bước 2 thân cây chính (RMIA) + kiểm "ConvNeXt chưa học xong ở vòng 30".
#   (a) 8 mô hình tham chiếu ConvNeXt-atto + 8 EfficientNet-B0 (học tập trung trên nửa train) ~50 phút
#   (b) FedAvg ConvNeXt 50 vòng × 3 seed ~45 phút
#   (c) rmia.py: chấm RMIA vs tấn công loss trên các mô hình nạn nhân (không cần GPU)
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in huan_luyen_tham_chieu.py rmia.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem1[0-9].sh" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy — đợi nó xong."; exit 1; }
for bb in convnext_atto efficientnet_b0; do for s in 0 1 2; do
  [ -f ketqua/lien_ket/fedavg_nguon_${bb}_txlp1_kt224_s${s}_logit.npz ] || { echo "LỖI: thiếu logit nạn nhân $bb seed $s"; exit 1; }
done; done
python -c "import pandas as pd; from du_lieu import doc_chia_tap; assert 'lesion_id' in doc_chia_tap().columns" \
  || { echo "LỖI: chia_tap.csv thiếu cột lesion_id"; exit 1; }

# ---------- 0. Ghi chép BAN_DO.md
moi_nhat() { ls -t "$D"/$1*.md 2>/dev/null | head -1; }
chen_truoc() {
  python - "$1" "$2" <<'PY'
import sys, pathlib
p = pathlib.Path("BAN_DO.md"); cu = p.read_text(encoding="utf-8")
moi = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"); moc = "\n" + sys.argv[2]
p.write_text(cu.replace(moc, "\n" + moi.rstrip() + "\n" + moc, 1) if moc in cu else cu + moi, encoding="utf-8")
PY
}
F11=$(moi_nhat BAN_DO_phan_them_sau_16-10)
if ! grep -q "Kết quả đêm 11" BAN_DO.md && [ -n "$F11" ] && grep -q "Kết quả đêm 11" "$F11"; then
  chen_truoc "$F11" "### Kết quả đêm 12"; echo "Đã ghi phần sau 16.10 (đến đêm 11)."; fi
F13=$(moi_nhat BAN_DO_dem13)
if ! grep -q "Kết quả đêm 13" BAN_DO.md && [ -n "$F13" ]; then cat "$F13" >> BAN_DO.md && echo "Đã ghi đêm 13."; fi
F14=$(moi_nhat BAN_DO_dem14)
if ! grep -q "Kết quả đêm 14" BAN_DO.md && [ -n "$F14" ]; then cat "$F14" >> BAN_DO.md && echo "Đã ghi đêm 14 + kế hoạch đêm 15."; fi
for k in "đêm 11" "đêm 12" "đêm 13" "đêm 14"; do grep -q "Kết quả $k" BAN_DO.md || echo "  (BAN_DO.md vẫn thiếu: Kết quả $k)"; done

# ---------- 1. Vá tên run khi số vòng khác 30 (để run 50 vòng không đè run 30 vòng). Làm một lần, có kiểm.
python - <<'PY' || { echo "LỖI: không vá được huan_luyen_lien_ket.py"; exit 1; }
import ast, pathlib, sys
p = pathlib.Path("huan_luyen_lien_ket.py"); s = p.read_text(encoding="utf-8")
if "{vg}" in s:
    print("huan_luyen_lien_ket.py: đã vá từ trước."); sys.exit(0)
cu = '    ten = f"{arg.thuat_toan}{mu}_{arg.kich_ban}_{arg.backbone}{db}{txl}{cb}{gh}_kt{arg.kt}_s{arg.seed}"'
assert s.count(cu) == 1, "không tìm thấy đúng dòng đặt tên"
moi = ('    vg = f"_v{arg.vong}" if arg.vong != 30 else ""   # số vòng khác mặc định -> tên riêng, không đè run cũ trên wandb\n'
       + cu.replace("{gh}_kt", "{gh}{vg}_kt"))
s = s.replace(cu, moi); ast.parse(s); p.write_text(s, encoding="utf-8"); print("huan_luyen_lien_ket.py: đã vá tên (_v50).")
PY

export HAMFL_GIA_THUYET="thiet_ke_B2,RMIA"
export HAMFL_DU_DOAN="(1) RMIA > loss: AUC tăng ≥0,03 và TPR@1%FPR tăng ≥2 lần ở cả ConvNeXt và EfficientNet. (2) ConvNeXt 50 vòng không hơn 30 vòng quá 0,02 → giữ 30 vòng"

# ---------- 2. Tự kiểm (~3–5 phút, im lặng): 1 mô hình tham chiếu học 1 epoch vào thư mục thử + 1 vòng FL 50 vòng
echo ">>> Tự kiểm 1/2: mô hình tham chiếu, 1 epoch (thư mục thử, không dùng kết quả)."
python huan_luyen_tham_chieu.py --backbone convnext_atto --cap 0 --nua 0 --epoch 1 --ra ketqua/tham_chieu_thu > dem15_thu.txt 2>&1
rc=$?; grep -v Warning dem15_thu.txt | tail -2
[ $rc -eq 0 ] && [ -f ketqua/tham_chieu_thu/tc_convnext_atto_txlp1_kt224_c0n0.npz ] \
  || { echo "LỖI: mô hình tham chiếu hỏng (mã $rc) — dán dem15_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm 2/2: 1 vòng FedAvg 50 vòng (lượt chính chạy tiếp từ đây)."
python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --vong 50 --seed 0 --dung-sau-vong 1 > dem15_thu2.txt 2>&1
rc=$?; grep -v Warning dem15_thu2.txt | grep -E "vòng|dừng|Error" | tail -2
[ $rc -eq 0 ] && [ -f ketqua/lien_ket/fedavg_nguon_convnext_atto_txlp1_v50_kt224_s0_ckpt.pt ] \
  || { echo "LỖI: chạy thử 50 vòng hỏng (mã $rc) — dán dem15_thu2.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong."

git add -A && git commit -q -m "dem15: RMIA (tham chieu + rmia.py), convnext 50 vong, ten _v{vong}, BAN_DO dem14" || true
cat > hang_doi_dem15.sh <<Q
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="$HAMFL_GIA_THUYET"
export HAMFL_DU_DOAN="$HAMFL_DU_DOAN"
for bb in convnext_atto efficientnet_b0; do
  for c in 0 1 2 3; do for n in 0 1; do
    python huan_luyen_tham_chieu.py --backbone \$bb --cap \$c --nua \$n
  done; done
done
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --backbone convnext_atto --kich-ban nguon --tien-xu-ly p1 --vong 50 --seed \$s
done
echo "=== XONG HUẤN LUYỆN ĐÊM 15 \$(date) ==="
python rmia.py > dem15_rmia.txt 2>&1; tail -25 dem15_rmia.txt
echo "=== XONG ĐÊM 15 \$(date) ==="
python bao_dong.py "Đêm 15 xong" "\$(sed -n '/Trung bình 3 seed/,\$p' dem15_rmia.txt | head -12)"
Q
setsid nohup bash hang_doi_dem15.sh > dem15.txt 2>&1 &
sleep 90; echo "--- 90 giây đầu:"; grep -v Warning dem15.txt | tail -3
