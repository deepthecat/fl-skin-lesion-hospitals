#!/usr/bin/env bash
# Đêm 14 — bước 1 thân cây chính: chọn thuật toán FL nền cho ConvNeXt-atto (LayerNorm).
# {FedProx μ=0,01, FedProx μ=0,1, FedAvgM β=0,9} × nguồn thật × 3 seed = 9 lượt (FedAvg đã có từ đêm 10). Ảnh 224, P1.
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
BB=convnext_atto
[ -f phan_tich_dem14.py ] || { echo "LỖI: thiếu phan_tich_dem14.py"; exit 1; }
pgrep -f "hang_doi_dem1[0-9].sh" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy — đợi nó xong."; exit 1; }
for s in 0 1 2; do
  [ -f ketqua/lien_ket/fedavg_nguon_${BB}_txlp1_kt224_s$s.json ] || { echo "LỖI: thiếu FedAvg ConvNeXt seed $s (mốc so sánh)"; exit 1; }
done
grep -q '"fedavgm"' huan_luyen_lien_ket.py && grep -q -- '--beta' huan_luyen_lien_ket.py \
  || { echo "LỖI: huan_luyen_lien_ket.py thiếu fedavgm/--beta"; exit 1; }

# ---------- 0. Ghi chép BAN_DO.md (mỗi phần một lần, giữ thứ tự thời gian)
moi_nhat() { ls -t "$D"/$1*.md 2>/dev/null | head -1; }
chen_truoc() {   # $1 = file cần chèn, $2 = mốc: chèn trước mốc nếu mốc đã có, không thì nối cuối
  python - "$1" "$2" <<'PY'
import sys, pathlib
p = pathlib.Path("BAN_DO.md"); cu = p.read_text(encoding="utf-8")
moi = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"); moc = "\n" + sys.argv[2]
p.write_text(cu.replace(moc, "\n" + moi.rstrip() + "\n" + moc, 1) if moc in cu else cu + moi, encoding="utf-8")
PY
}
F11=$(moi_nhat BAN_DO_phan_them_sau_16-10)
if ! grep -q "Kết quả đêm 11" BAN_DO.md; then
  if [ -n "$F11" ] && grep -q "Kết quả đêm 11" "$F11"; then chen_truoc "$F11" "### Kết quả đêm 12"; echo "Đã ghi phần sau 16.10 (đến đêm 11)."
  else echo "CẢNH BÁO: vẫn chưa có BAN_DO_phan_them_sau_16-10*.md trong Research_2026_Code (không ảnh hưởng việc chạy)."; fi
fi
F13=$(moi_nhat BAN_DO_dem13)
if ! grep -q "Kết quả đêm 13" BAN_DO.md; then
  if [ -n "$F13" ]; then cat "$F13" >> BAN_DO.md && echo "Đã ghi đêm 13 + tái cấu trúc."
  else echo "CẢNH BÁO: chưa có BAN_DO_dem13.md trong Research_2026_Code."; fi
fi

export HAMFL_GIA_THUYET="G9,thiet_ke_B1"
export HAMFL_DU_DOAN="ConvNeXt (LayerNorm): FedProx ≈ FedAvg (trôi không phải nút nghẽn, như EfficientNet); FedAvgM ≈ FedAvg hoặc kém hơn. Không thuật toán nào đạt luật chọn → giữ FedAvg"

# ---------- 1. Tự kiểm: FedAvgM chưa từng chạy trên máy này → chạy thật 1 vòng (lượt chính sẽ chạy tiếp từ đây)
echo ">>> Tự kiểm: chạy thử 1 vòng FedAvgM (~2–4 phút, im lặng)."
python huan_luyen_lien_ket.py --thuat-toan fedavgm --backbone $BB --kich-ban nguon --tien-xu-ly p1 --seed 0 --dung-sau-vong 1 > dem14_thu.txt 2>&1
rc=$?
grep -v Warning dem14_thu.txt | grep -E "vòng|dừng thử|Error|Traceback" | tail -3
[ $rc -eq 0 ] && [ -f ketqua/lien_ket/fedavgm_nguon_${BB}_txlp1_kt224_s0_ckpt.pt ] \
  || { echo "LỖI: chạy thử FedAvgM hỏng (mã $rc) — dán dem14_thu.txt cho Claude"; exit 1; }
v1=$(grep -oP 'vòng +1/30 .*val bacc \K[0-9.]+' dem14_thu.txt | head -1)
echo ">>> FedAvgM vòng 1: val bacc $v1. Vòng 1 chưa có đà nên giống FedAvg — bước này chỉ kiểm code chạy được; nếu đà làm vỡ thì sẽ thấy trong đường val từ vòng 2 trở đi (xem trên wandb)."
echo ">>> Tự kiểm xong."

git add -A && git commit -q -m "dem14: convnext fedprox/fedavgm + BAN_DO dem13 + tai cau truc" || true
cat > hang_doi_dem14.sh <<Q
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
export HAMFL_GIA_THUYET="$HAMFL_GIA_THUYET"
export HAMFL_DU_DOAN="$HAMFL_DU_DOAN"
for s in 0 1 2; do
  python huan_luyen_lien_ket.py --thuat-toan fedavgm --backbone $BB --kich-ban nguon --tien-xu-ly p1 --seed \$s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --backbone $BB --kich-ban nguon --tien-xu-ly p1 --seed \$s
  python huan_luyen_lien_ket.py --thuat-toan fedprox --mu 0.1 --backbone $BB --kich-ban nguon --tien-xu-ly p1 --seed \$s
done
echo "=== XONG ĐÊM 14 \$(date) ==="
python bao_dong.py "Đêm 14 xong" "\$(python phan_tich_dem14.py 2>&1 | sed -n '3,14p')"
Q
setsid nohup bash hang_doi_dem14.sh > dem14.txt 2>&1 &
sleep 90; echo "--- 90 giây đầu:"; grep -v Warning dem14.txt | tail -3
