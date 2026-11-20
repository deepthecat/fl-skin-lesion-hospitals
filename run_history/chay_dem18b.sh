#!/usr/bin/env bash
# Đêm 18b (ngắn, ~25 phút): (1) chạy lại mô phỏng máy chủ với không-thành-viên = val CỦA CHÍNH bệnh viện, ở 2 mức lr;
# (2) đối chiếu Fed-ISIC2019 của mình với số gốc của FLamby (phần 3 phan_tich_isic.py, không cần GPU).
set -u
cd ~/ham_fl
source .venv/bin/activate
for f in sim_may_chu.py phan_tich_isic.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
grep -q "val_k" sim_may_chu.py || { echo "LỖI: sim_may_chu.py chưa phải bản 2 — chép lại từ Research_2026_Code"; exit 1; }
pgrep -f "hang_doi_dem1[0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
git add -A && git commit -q -m "dem18b: sim_may_chu v2 (val cung benh vien, 2 muc lr)" || true
echo "=== Fed-ISIC2019: mình vs FLamby"
python phan_tich_isic.py 2>&1 | sed -n '/^3\./,$p'
cat > hang_doi_dem18b.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
python sim_may_chu.py > dem18b_may_chu.txt 2>&1
echo "=== XONG ĐÊM 18b $(date) ==="
Q
setsid nohup bash hang_doi_dem18b.sh > dem18b.txt 2>&1 &
echo ">>> Mô phỏng máy chủ đang chạy nền (~25 phút). Xem kết quả: sed -n '/Mức lộ/,\$p' dem18b_may_chu.txt"
