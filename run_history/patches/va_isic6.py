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
