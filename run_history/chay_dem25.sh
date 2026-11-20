#!/usr/bin/env bash
# Lượt 25 (~3,5–4 giờ): Fed-ISIC — tập trung 60 epoch (cùng ngân sách với FedAvg 27×100) cho EfficientNet-B0 và ConvNeXt-atto, công thức HAM, 3 seed → khoảng tập trung − FL công bằng.
set -u
cd ~/ham_fl
source .venv/bin/activate
for f in fed_isic.py du_lieu.py bao_dong.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
cat > va_isic5.py <<'PY'
"""va_isic5.py — thêm _e{epoch} vào tên lượt tập trung/cục bộ khi epoch khác 20 (để lượt 60 epoch không bị bỏ qua vì 'đã có'). Chạy lại vô hại."""
from pathlib import Path
f = Path("fed_isic.py"); s = f.read_text()
cu = 'vbt = f"_v{arg.vong}b{arg.so_buoc}" if (arg.vong or arg.so_buoc != 100) else ""\n'
moi = cu + '    vbt += f"_e{arg.epoch}" if (arg.che_do in ("tap_trung", "cuc_bo") and arg.epoch != 20) else ""\n'
if '_e{arg.epoch}' in s:
    print("va_isic5: đã vá từ trước.")
elif cu in s:
    f.write_text(s.replace(cu, moi)); print("va_isic5: đã vá tên (thêm _e{epoch} cho tập trung).")
else:
    raise SystemExit("va_isic5: KHÔNG tìm thấy dòng vbt — dán lỗi này cho Claude.")
PY
cat > phan_tich_isic7.py <<'PY'
"""
phan_tich_isic7.py — Lượt 25: khoảng tập trung − FedAvg trên Fed-ISIC khi CẢ HAI cùng ngân sách dài (tập trung 60 epoch ≈ FedAvg 27 vòng × 100 bước × 6 trung tâm).
Công thức HAM, bAcc test TB đều 6 trung tâm, 3 seed.
Luật viết trước: "ConvNeXt thiệt ít hơn khi FL" đúng nếu khoảng(EffNet) − khoảng(ConvNeXt) > 0,02 VÀ dương ở cả 3 seed (ghép theo seed).
"""
import json
import warnings

import numpy as np

from du_lieu import RA

warnings.filterwarnings("ignore")
TT = ["BCN", "molemax", "modern", "rosendahl", "MSK", "dias"]
MH = [("EfficientNet-B0", "efficientnet_b0"), ("ConvNeXt-atto", "convnext_atto")]


def doc(ten, k="test_tb_tram"):
    x = []
    for s in range(3):
        f = RA / "fed_isic" / f"{ten}_s{s}.json"
        x.append(json.load(open(f))["ket_qua"][k] if f.exists() else np.nan)
    return np.array(x, float)


def main():
    print("Fed-ISIC2019 — công thức HAM, bAcc test TB đều 6 trung tâm (3 seed)\n")
    print(f"{'mô hình':<17}{'TT 20ep':>9}{'TT 60ep':>9}{'FedAvg 27×100':>15}{'khoảng (60ep)':>15}   khoảng theo seed")
    k = {}
    for nhan, bb in MH:
        t20 = doc(f"isic_tap_trung_cc_{bb}_ham"); t60 = doc(f"isic_tap_trung_cc_{bb}_ham_e60")
        fa = doc(f"isic_fedavg_cc_{bb}_ham_v27b100"); k[bb] = t60 - fa
        print(f"{nhan:<17}{np.nanmean(t20):>9.3f}{np.nanmean(t60):>9.3f}{np.nanmean(fa):>15.3f}{np.nanmean(k[bb]):>15.3f}   {[f'{v:.3f}' for v in k[bb]]}")
    print("\nTheo trung tâm, tập trung 60 epoch (EffNet / ConvNeXt):")
    for t in TT:
        print(f"  {t:<10}{np.nanmean(doc('isic_tap_trung_cc_efficientnet_b0_ham_e60', 'test_' + t)):.3f} / "
              f"{np.nanmean(doc('isic_tap_trung_cc_convnext_atto_ham_e60', 'test_' + t)):.3f}")
    d = k["efficientnet_b0"] - k["convnext_atto"]
    if np.isnan(d).any():
        print("\nTHIẾU lượt — chưa kết luận."); return
    ok = d.mean() > 0.02 and (d > 0).all()
    print(f"\nkhoảng EffNet − khoảng ConvNeXt: {d.mean():+.3f}  theo seed {[f'{v:+.3f}' for v in d]}")
    print("KẾT LUẬN theo luật viết trước:",
          "ĐÚNG — cùng ngân sách dài, ConvNeXt (LayerNorm) thiệt ít hơn khi FL; thiết kế 1 lặp lại trên bộ thứ hai." if ok else
          "KHÔNG được ủng hộ — ConvNeXt hơn ở FedAvg nhưng tập trung cũng hơn tương ứng; khoảng FL không nhỏ hơn rõ.")


if __name__ == "__main__":
    main()
PY
python va_isic5.py || exit 1
grep -q "Kết quả lượt 24" BAN_DO.md || cat >> BAN_DO.md <<'MD'
### Kết quả lượt 24 (26.10.2026) — Fed-ISIC, FedAvg 27 vòng × 100 bước (gấp 3 tổng bước), công thức HAM, bAcc test TB đều 6 trung tâm, 3 seed
| lịch FL | EfficientNet-B0 | ConvNeXt-atto | ConvNeXt − EffNet (theo seed) |
|---|---|---|---|
| 9 × 100 | 0,643 | 0,628 | −0,015 |
| 27 × 33 (cùng tổng bước) | 0,638 | 0,639 | +0,001 |
| 27 × 100 (×3 bước) | 0,640 | 0,683 | +0,043 (+0,060 / +0,026 / +0,044) |
- ĐO ĐƯỢC: theo luật viết trước, "ConvNeXt thắng khi FL học lâu hơn" ĐÚNG (> 0,02, cả 3 seed dương). EfficientNet KHÔNG lên khi học lâu hơn (0,643 / 0,638 / 0,640 — kịch trần sớm), ConvNeXt lên đều (0,628 → 0,639 → 0,683). Cùng dáng với HAM (đêm 14: EfficientNet kịch trần vòng ~10, ConvNeXt lên tới vòng 25–29). Khớp Siomos (TMLR 2025).
- Theo trung tâm (EffNet / ConvNeXt): dias 0,697 / 0,811, modern 0,750 / 0,805, molemax 0,698 / 0,767; BCN 0,742 / 0,731.
- SỬA kết luận đêm 22–23: lợi thế ConvNeXt KHÔNG chỉ có trên HAM; nó bị giấu bởi ngân sách huấn luyện ngắn kiểu FLamby. Bài học phương pháp: ngân sách ngắn có thể đổi thứ hạng kiến trúc.
- CHƯA CÔNG BẰNG để nói về KHOẢNG: FedAvg 27×100 học gấp 3 bước so với tập trung 20 epoch → lượt 25 chạy tập trung 60 epoch.
### Lượt 25 (26.10.2026): tập trung 60 epoch (≈ cùng tổng bước với FedAvg 27×100), công thức HAM, 2 mô hình × 3 seed. Luật viết trước: "ConvNeXt thiệt ít hơn khi FL" đúng nếu khoảng(EffNet) − khoảng(ConvNeXt) > 0,02 và dương cả 3 seed. Vá fed_isic.py: tên tập trung thêm _e{epoch} khi khác 20.
MD
export HAMFL_GIA_THUYET="thiet_ke_1,du_lieu_thu_hai,hoc_lau"
export HAMFL_DU_DOAN="cùng ngân sách dài: khoảng tập trung − FedAvg của ConvNeXt nhỏ hơn EfficientNet > 0,02, cả 3 seed"
echo ">>> Tự kiểm: ConvNeXt tập trung 1 epoch, 50 ảnh/trung tâm (thư mục thử)."
python fed_isic.py --che-do tap_trung --backbone convnext_atto --cong-thuc ham --epoch 1 --gioi-han-anh 50 --workers 2 --ra ketqua/fed_isic_thu > dem25_thu.txt 2>&1
rc=$?; grep -E "epoch|Đã ghi|Error|Traceback" dem25_thu.txt | tail -2
[ $rc -eq 0 ] && grep -q "_e1_" dem25_thu.txt || { echo "LỖI: tự kiểm hỏng (mã $rc) — dán dem25_thu.txt cho Claude"; exit 1; }
echo ">>> Tự kiểm xong (tên có _e1 → vá đúng)."
git add -A && git commit -q -m "luot25: Fed-ISIC tap trung 60 epoch" || true
cat > hang_doi_dem25.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for s in 0 1 2; do for bb in convnext_atto efficientnet_b0; do
  python fed_isic.py --che-do tap_trung --backbone $bb --cong-thuc ham --epoch 60 --workers 2 --seed $s
done; done
echo "=== XONG LƯỢT 25 $(date) ==="
python phan_tich_isic7.py > dem25_phan_tich.txt 2>&1; cat dem25_phan_tich.txt
cp dem25_phan_tich.txt /mnt/c/Users/admin/Downloads/Research_2026_Code/ 2>/dev/null
python bao_dong.py "Lượt 25 xong" "$(head -6 dem25_phan_tich.txt)"
Q
setsid nohup bash hang_doi_dem25.sh > dem25.txt 2>&1 &
sleep 150; echo "--- 2,5 phút đầu:"; grep -v Warning dem25.txt | grep -E "epoch|Error|Traceback" | tail -2; free -h | head -2
echo ">>> Thấy 'epoch  1/60' là ổn — đóng VS Code được."
