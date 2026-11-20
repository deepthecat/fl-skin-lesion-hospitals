#!/usr/bin/env bash
# Đêm 27 (~2–3,5 giờ): THIẾT KẾ ĐỀ XUẤT — học trước EfficientNet-B0 trên ảnh da CÔNG KHAI của bệnh viện khác (BCN, ISIC 2019),
# đóng băng, rồi DP lớp cuối trong FL 4 bệnh viện HAM (ε ∈ {inf, clip, 8, 3, 1, 0.5} × lr {16, 4, 1} × 3 seed). Tự chứa.
set -u
cd ~/ham_fl
source .venv/bin/activate
D=/mnt/c/Users/admin/Downloads/Research_2026_Code
for f in fed_isic.py dp_tuyen_tinh.py du_lieu.py kiem_trung_test.py bao_dong.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
[ -d ~/du_lieu_ngoai/isic2019_raw/isic2019_224_goc ] || { echo "LỖI: thiếu ảnh ISIC bản goc (~/du_lieu_ngoai/isic2019_raw/isic2019_224_goc)"; exit 1; }
pgrep -f "hang_doi_dem[12][0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
python -c "import torch; torch.ones(10,device='cuda'); print('GPU ổn')" || { echo "LỖI: GPU không phản hồi — PowerShell: wsl --shutdown"; exit 1; }
cat > va_isic6.py <<'PY'
"""va_isic6.py — fed_isic.py: thêm --loai-tru (CSV cột image: bỏ các ảnh này khỏi train) và --luu-mo-hinh (lưu trọng số cuối vào ketqua/mo_hinh/). Chạy lại vô hại."""
from pathlib import Path
f = Path("fed_isic.py"); s = f.read_text()
if "--luu-mo-hinh" in s:
    print("va_isic6: đã vá từ trước."); raise SystemExit(0)
thay = [
    ('    ap.add_argument("--seed", type=int, default=0)\n',
     '    ap.add_argument("--seed", type=int, default=0)\n'
     '    ap.add_argument("--loai-tru", default="", help="CSV cột image: bỏ khỏi train (vd. ảnh nghi trùng HAM)")\n'
     '    ap.add_argument("--luu-mo-hinh", action="store_true", help="lưu trọng số cuối vào ketqua/mo_hinh/<tên>.pt")\n'),
    ('    tr, te = cs[cs.phan == "train"].reset_index(drop=True), cs[cs.phan == "test"].reset_index(drop=True)\n',
     '    tr, te = cs[cs.phan == "train"].reset_index(drop=True), cs[cs.phan == "test"].reset_index(drop=True)\n'
     '    if arg.loai_tru:\n'
     '        bo = set(pd.read_csv(arg.loai_tru).image.astype(str)); n0 = len(tr)\n'
     '        tr = tr[~tr.image.astype(str).isin(bo)].reset_index(drop=True); print(f"Loại trừ {n0 - len(tr)} ảnh train theo {arg.loai_tru}")\n'),
    ('    vbt += f"_e{arg.epoch}" if (arg.che_do in ("tap_trung", "cuc_bo") and arg.epoch != 20) else ""\n',
     '    vbt += f"_e{arg.epoch}" if (arg.che_do in ("tap_trung", "cuc_bo") and arg.epoch != 20) else ""\n'
     '    vbt += "_lt" if arg.loai_tru else ""\n'),
    ('    f_ck.unlink(missing_ok=True)\n',
     '    if arg.luu_mo_hinh:\n'
     '        (RA / "mo_hinh").mkdir(exist_ok=True); torch.save(mh.state_dict(), RA / "mo_hinh" / f"{ten}.pt")\n'
     '        print(f"Đã lưu trọng số: {RA / \'mo_hinh\' / (ten + \'.pt\')}")\n'
     '    f_ck.unlink(missing_ok=True)\n'),
]
for cu, moi in thay:
    if s.count(cu) != 1:
        raise SystemExit(f"va_isic6: không tìm thấy đúng 1 chỗ:\n{cu}— dán lỗi này cho Claude.")
    s = s.replace(cu, moi)
f.write_text(s); print("va_isic6: đã vá (--loai-tru, --luu-mo-hinh, tên thêm _lt).")
PY
cat > va_dptt2.py <<'PY'
"""va_dptt2.py — dp_tuyen_tinh.py: thêm --trong-so-backbone (nạp xương sống đã học trước, vd. trên ảnh BCN công khai) và --nhan-dt (nhãn gắn vào tên file đặc trưng + tên lượt). Chạy lại vô hại."""
from pathlib import Path
f = Path("dp_tuyen_tinh.py"); s = f.read_text()
if "--trong-so-backbone" in s:
    print("va_dptt2: đã vá từ trước."); raise SystemExit(0)
thay = [
    ('def dac_trung(ten, kt, ds, tb, pretrained=True, kieu="p0"):\n',
     'def dac_trung(ten, kt, ds, tb, pretrained=True, kieu="p0", ts="", nhan=""):\n'),
    ('    duoi = "" if kieu == "p0" else f"_{kieu}"\n',
     '    duoi = "" if kieu == "p0" else f"_{kieu}"\n    duoi += f"_{nhan}" if nhan else ""\n'),
    ('    mh.reset_classifier(0)',
     '    if ts:                                      # xương sống học trước trên dữ liệu công khai (bỏ tầng phân loại cũ)\n'
     '        sd = {k: v for k, v in torch.load(ts, map_location="cpu").items() if not k.startswith(("classifier.", "head.fc.", "fc."))}\n'
     '        r = mh.load_state_dict(sd, strict=False)\n'
     '        thieu = [k for k in r.missing_keys if not k.startswith(("classifier.", "head.fc.", "fc."))]\n'
     '        print(f"Nạp {ts}: thiếu {len(thieu)} khoá, thừa {len(r.unexpected_keys)} khoá", flush=True)\n'
     '        if thieu or r.unexpected_keys:\n'
     '            raise SystemExit(f"LỖI nạp xương sống: thiếu {thieu[:5]} thừa {r.unexpected_keys[:5]}")\n'
     '    mh.reset_classifier(0)'),
    ('kieu=arg.tien_xu_ly)', 'kieu=arg.tien_xu_ly, ts=arg.trong_so_backbone, nhan=arg.nhan_dt)'),
    ('_{arg.backbone}{txl}', "_{arg.backbone}{'_' + arg.nhan_dt if arg.nhan_dt else ''}{txl}"),
    ('    ap.add_argument("--seed", type=int, default=0)\n',
     '    ap.add_argument("--seed", type=int, default=0)\n'
     '    ap.add_argument("--trong-so-backbone", default="", help="file .pt trọng số xương sống (timm) học trước")\n'
     '    ap.add_argument("--nhan-dt", default="", help="nhãn đặc trưng, vd. bcn")\n'),
]
for cu, moi in thay:
    if s.count(cu) != 1:
        raise SystemExit(f"va_dptt2: không tìm thấy đúng 1 chỗ:\n{cu}\n— dán lỗi này cho Claude.")
    s = s.replace(cu, moi)
f.write_text(s); print("va_dptt2: đã vá (--trong-so-backbone, --nhan-dt).")
PY
cat > kiem_trung_bcn.py <<'PY'
"""
kiem_trung_bcn.py — trước khi dùng ảnh BCN (ISIC 2019, trung tâm 0) làm dữ liệu CÔNG KHAI để học trước, kiểm không có
ảnh nào trùng với BẤT KỲ ảnh HAM10000 nào (train / val / test). Cùng cách hai tầng như kiem_trung_test.py:
  1. ứng viên: pHash cách nhau ≤ 10 bit;  2. xác nhận: tương quan chi tiết ≥ 0,80 (mức sàn của kiem_trung_test.py).
Ghi ketqua/bcn_loai_tru.csv (cột image) = ảnh BCN bị loại (đã xác nhận trùng). Ứng viên chưa xác nhận chỉ báo, không loại.
Lý do: dữ liệu "công khai" mà chứa bản sao ảnh của bệnh nhân HAM thì bảo đảm DP về sau không còn đúng nghĩa.
"""
import time

import numpy as np
import pandas as pd

from du_lieu import RA, doc_chia_tap
from fed_isic import ANH
from kiem_trung_test import dac_trung

NGUONG = 0.80
LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def main():
    t0 = time.time()
    ds = doc_chia_tap()
    cs = pd.read_csv(RA / "fed_isic_chia.csv")
    bcn = cs[(cs.center == 0) & (cs.phan == "train")].reset_index(drop=True)
    th = ANH["goc"]
    print(f"HAM10000: {len(ds)} ảnh · BCN train: {len(bcn)} ảnh ({th})", flush=True)
    dh = [dac_trung(p) for p in ds.duong_dan]
    Hh = np.array([x[0] for x in dh], np.uint64); Vh = np.stack([x[1] for x in dh])
    db = [dac_trung(th / f"{t}.jpg") for t in bcn.image]
    Hb = np.array([x[0] for x in db], np.uint64); Vb = np.stack([x[1] for x in db])
    print(f"Đặc trưng xong: {time.time() - t0:.0f}s", flush=True)
    hang = []
    for i0 in range(0, len(Hb), 256):
        x = (Hb[i0:i0 + 256, None] ^ Hh[None, :]).view(np.uint8).reshape(-1, len(Hh), 8)
        d = LUT[x].sum(-1)                                       # khoảng cách Hamming, (lô × số ảnh HAM)
        for a, b in zip(*np.nonzero(d <= 10)):
            hang.append(dict(image=bcn.image.iat[i0 + a], anh_ham=ds.image.iat[b], phan_ham=ds.phan.iat[b],
                             bit=int(d[a, b]), tuong_quan=float(Vb[i0 + a] @ Vh[b])))
    kq = pd.DataFrame(hang, columns=["image", "anh_ham", "phan_ham", "bit", "tuong_quan"])
    xn = kq[kq.tuong_quan >= NGUONG]
    xn[["image"]].drop_duplicates().to_csv(RA / "bcn_loai_tru.csv", index=False)
    kq.sort_values("tuong_quan", ascending=False).to_csv(RA / "bcn_trung_ung_vien.csv", index=False)
    print(f"Cặp ứng viên (≤ 10 bit): {len(kq)} · ảnh BCN ứng viên: {kq.image.nunique()}")
    print(f"XÁC NHẬN TRÙNG (tương quan ≥ {NGUONG}): {xn.image.nunique()} ảnh BCN → loại khỏi dữ liệu công khai")
    if len(kq):
        print("Tương quan cao nhất:", kq.tuong_quan.max().round(3))
    print(f"Đã ghi {RA / 'bcn_loai_tru.csv'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
PY
cat > phan_tich_dp_cong_khai.py <<'PY'
"""
phan_tich_dp_cong_khai.py — Đêm 27: thiết kế đề xuất. DP lớp cuối (L0) trong FL 4 bệnh viện HAM, xương sống EfficientNet-B0
ĐÓNG BĂNG, so hai nguồn đặc trưng: ImageNet (thiết kế 2 hiện tại) vs học trước trên ảnh da CÔNG KHAI của bệnh viện khác (BCN, ISIC 2019,
đã loại ảnh nghi trùng HAM). Với mỗi ε: lr tốt nhất trên val trong {16, 4, 1}; TB 3 seed.
Luật viết trước: "thiết kế cân bằng" nếu đặc trưng BCN ở ε=3 đạt val ≥ 0,55 (ImageNet: ~0,43).
"""
import json
import warnings

import numpy as np
import pandas as pd

from du_lieu import RA

warnings.filterwarnings("ignore")
MUC = ("inf", "8", "3", "1")
NGUON = {"ImageNet (thiết kế 2)": ({"16": "dp_tuyen_tinh", "4": "dp_tuyen_tinh_lr4", "1": "dp_tuyen_tinh_lr1"}, ""),
         "BCN công khai (đề xuất)": ({"16": "dp_tuyen_tinh_bcn_lr16", "4": "dp_tuyen_tinh_bcn_lr4", "1": "dp_tuyen_tinh_bcn_lr1"}, "_bcn")}


def main():
    hang = []
    for nguon, (thu, nh) in NGUON.items():
        for lr, d in thu.items():
            for m in MUC:
                for s in range(3):
                    f = RA / d / f"dptt_nguon_eps{m}_efficientnet_b0{nh}_txlp1_kt224_s{s}.json"
                    if f.exists():
                        k = json.load(open(f))["ket_qua"]
                        hang.append(dict(nguon=nguon, lr=lr, eps=m, seed=s, val=k["val"]["bacc"], test=k["test"]["bacc"]))
    if not hang:
        raise SystemExit("Chưa có kết quả.")
    b = pd.DataFrame(hang)
    g = b.groupby(["nguon", "eps", "lr"]).agg(val=("val", "mean"), test=("test", "mean"), sd=("val", "std"), n=("val", "count")).reset_index()
    tot = g.loc[g.groupby(["nguon", "eps"]).val.idxmax()]
    pd.set_option("display.width", 200)
    print("FL 4 bệnh viện HAM, DP lớp cuối, EfficientNet-B0 đóng băng — val bAcc (lr tốt nhất cho từng ε, TB 3 seed)\n")
    print(tot.pivot(index="nguon", columns="eps", values="val")[list(MUC)].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTest bAcc tương ứng:\n")
    print(tot.pivot(index="nguon", columns="eps", values="test")[list(MUC)].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nlr được chọn / số seed:\n")
    print(tot.assign(x=tot.lr + " / " + tot.n.astype(str)).pivot(index="nguon", columns="eps", values="x")[list(MUC)].to_string())
    try:
        v3 = tot[(tot.nguon == "BCN công khai (đề xuất)") & (tot.eps == "3")].val.iat[0]
        v3i = tot[(tot.nguon == "ImageNet (thiết kế 2)") & (tot.eps == "3")].val.iat[0]
    except IndexError:
        print("\nTHIẾU lượt ε=3 — chưa kết luận."); return
    print(f"\nε=3: BCN {v3:.3f} vs ImageNet {v3i:.3f} (chênh {v3 - v3i:+.3f}). Tham khảo không DP, học toàn bộ: EfficientNet FedAvg 0,661 · ConvNeXt FedAvg 0,741.")
    print("KẾT LUẬN theo luật viết trước:",
          "ĐẠT — có thiết kế cân bằng (ε=3, val ≥ 0,55)." if v3 >= 0.55 else
          "KHÔNG đạt 0,55 — kết luận đánh đổi đứng, kèm mức mà đòn bẩy 'đặc trưng tốt hơn' kéo được.")


if __name__ == "__main__":
    main()
PY
python va_isic6.py && python va_dptt2.py || exit 1
python -c "import ast; [ast.parse(open(f).read()) for f in ('fed_isic.py','dp_tuyen_tinh.py')]" || { echo "LỖI: vá hỏng cú pháp"; exit 1; }
if ! grep -q "Kết quả lượt 26" BAN_DO.md && [ -f "$D/BAN_DO_dem26.md" ]; then cat "$D/BAN_DO_dem26.md" >> BAN_DO.md && echo "Đã ghi lượt 26 vào BAN_DO.md."; fi
grep -q "Đêm 27 (27.10.2026)" BAN_DO.md || cat >> BAN_DO.md <<'MD'
### Đêm 27 (27.10.2026): THIẾT KẾ ĐỀ XUẤT — đặc trưng da liễu từ dữ liệu CÔNG KHAI cho DP lớp cuối
- Lý do (Deep 27.10): kết luận "đánh đổi, không cân bằng được" yếu vì mới thử DP với MỘT bộ đặc trưng (ImageNet). Đòn bẩy chính trong tài liệu DP là đặc trưng tốt hơn (Tramèr & Boneh, ICLR 2021 — kiểm bản chính thức trước khi trích).
- Thiết kế: học trước EfficientNet-B0 (timm, công thức HAM, 20 epoch) trên ảnh BCN train của ISIC 2019 (≈ 9,9k ảnh, công khai, bệnh viện khác, KHÔNG thuộc 4 nguồn HAM), sau khi loại ảnh xác nhận trùng với bất kỳ ảnh HAM nào (kiem_trung_bcn.py: pHash ≤ 10 bit + tương quan chi tiết ≥ 0,80). Đóng băng; 4 bệnh viện HAM học tầng cuối bằng DP-SGD (giữ nguyên dp_tuyen_tinh, nguồn thật, P1).
- Luật viết trước: "thiết kế cân bằng" nếu ε=3 val ≥ 0,55 (ImageNet ~0,43). Không đạt → kết luận đánh đổi đứng, kèm mức đòn bẩy kéo được. Tối đa 3 đêm cho nhánh này.
- Giới hạn biết trước: xương sống học trước chỉ 1 seed; BCN dùng ảnh 'goc' (không cân màu) để khớp P1 của HAM.
MD
export HAMFL_GIA_THUYET="thiet_ke_de_xuat,DP,dac_trung_cong_khai"
export HAMFL_DU_DOAN="đặc trưng BCN: DP lớp cuối ε=3 val ≥ 0,55 (ImageNet ~0,43)"
echo ">>> Tự kiểm: học trước BCN 1 epoch / 50 ảnh, lưu trọng số, nạp lại vào xương sống HAM."
rm -f ketqua/fed_isic_thu/isic_cuc_bo_tram0_goc_efficientnet_b0_ham_e1_s0*
python fed_isic.py --che-do cuc_bo --tram 0 --anh goc --backbone efficientnet_b0 --cong-thuc ham --epoch 1 --gioi-han-anh 50 --workers 2 --luu-mo-hinh --ra ketqua/fed_isic_thu > dem27_thu.txt 2>&1
rc=$?; grep -E "Đã lưu|Error|Traceback" dem27_thu.txt | tail -2
[ $rc -eq 0 ] || { echo "LỖI: tự kiểm học trước hỏng (mã $rc) — dán dem27_thu.txt cho Claude"; exit 1; }
python - <<'PY' || { echo "LỖI: nạp trọng số vào xương sống HAM hỏng — dán lỗi cho Claude"; exit 1; }
import timm, torch
from pathlib import Path
from du_lieu import RA, LOP
f = RA / "mo_hinh" / "isic_cuc_bo_tram0_goc_efficientnet_b0_ham_e1_s0.pt"
mh = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(LOP))
sd = {k: v for k, v in torch.load(f, map_location="cpu").items() if not k.startswith(("classifier.", "head.fc.", "fc."))}
r = mh.load_state_dict(sd, strict=False)
thieu = [k for k in r.missing_keys if not k.startswith("classifier.")]
assert not thieu and not r.unexpected_keys, (thieu[:5], r.unexpected_keys[:5])
f.unlink(); print("Nạp lại trọng số: khớp đủ khoá.")
PY
echo ">>> Tự kiểm xong."
git add -A && git commit -q -m "dem27: thiet ke de xuat - dac trung BCN cong khai + DP lop cuoi" || true
cat > hang_doi_dem27.sh <<'Q'
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
Q
setsid nohup bash hang_doi_dem27.sh > dem27.txt 2>&1 &
sleep 150; echo "--- 2,5 phút đầu:"; grep -v Warning dem27.txt | tail -3; free -h | head -2
echo ">>> Thấy dòng '=== (1) kiểm trùng' hoặc 'HAM10000: ... ảnh' là ổn — đi làm việc khác được."
