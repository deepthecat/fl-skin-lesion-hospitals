exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
echo "=== (1) kiểm trùng BCN ↔ HAM $(date)"; python kiem_trung_bcn.py || exit 1
echo "=== (2) học trước trên BCN $(date)"
python fed_isic.py --che-do cuc_bo --tram 0 --anh goc --backbone efficientnet_b0 --cong-thuc ham --loai-tru ketqua/bcn_loai_tru.csv --luu-mo-hinh --workers 2 --seed 0
TS=ketqua/mo_hinh/isic_cuc_bo_tram0_goc_efficientnet_b0_ham_lt_s0.pt
[ -f "$TS" ] || { echo "LỖI: không có $TS"; exit 1; }
echo "=== (3) DP lớp cuối trên đặc trưng BCN $(date)"
for lr in 4 1 16; do
  python dp_tuyen_tinh.py --backbone efficientnet_b0 --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr $lr --trong-so-backbone $TS --nhan-dt bcn --ra ketqua/dp_tuyen_tinh_bcn_lr$lr
done
echo "=== XONG ĐÊM 27 $(date) ==="
python phan_tich_dp_cong_khai.py > dem27_phan_tich.txt 2>&1; cat dem27_phan_tich.txt
grep -E "Cặp ứng viên|XÁC NHẬN" dem27.txt >> dem27_phan_tich.txt
cp dem27_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Đêm 27 xong" "$(head -6 dem27_phan_tich.txt)"
