#!/usr/bin/env bash
# Vẽ hình 2–4 của paper từ file kết quả đã chạy, rồi chép sang Research_2026_Code/hinh_paper_tu_ketqua.
set -u
cd ~/ham_fl && source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
cp "$D/ve_hinh_paper.py" . && python ve_hinh_paper.py | tee hinh_paper_so_lieu.txt
mkdir -p "$D/hinh_paper_tu_ketqua" && cp ketqua/hinh_paper/* hinh_paper_so_lieu.txt "$D/hinh_paper_tu_ketqua/" && echo "xong — đã chép sang hinh_paper_tu_ketqua"
