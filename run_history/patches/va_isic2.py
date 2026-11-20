"""va_isic2.py — thêm --kich-ban nguon|tron cho fed_isic.py (FL dùng cột may_tron để chia client). Chạy lại nhiều lần an toàn."""
import ast
import pathlib
import sys

p = pathlib.Path("fed_isic.py"); s = p.read_text(encoding="utf-8")
if "--kich-ban" in s:
    print("fed_isic.py: đã vá kịch bản từ trước."); sys.exit(0)
if "--toi-uu-fl" not in s:
    sys.exit("LỖI: chạy va_isic.py trước (cần --backbone/--toi-uu-fl)")
thay = [
    ('    ap.add_argument("--toi-uu-fl",',
     '    ap.add_argument("--kich-ban", default="nguon", choices=("nguon", "tron"), help="client = trung tâm thật | trộn máy (cột may_tron)")\n'
     '    ap.add_argument("--toi-uu-fl",'),
    ('    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}_s{arg.seed}"\n',
     '    kbt = "" if arg.kich_ban == "nguon" else f"_{arg.kich_ban}"\n'
     '    if arg.kich_ban == "tron" and arg.che_do not in ("fedavg", "fedprox"):\n'
     '        raise SystemExit("--kich-ban tron chỉ dùng cho fedavg/fedprox")\n'
     '    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{kbt}_s{arg.seed}"\n'),
    ('        tram = sorted(int(c) for c in tr.center.unique())\n',
     '        kh = tr.center.values if arg.kich_ban == "nguon" else tr.may_tron.values.astype(int)   # client của từng ảnh train\n'
     '        tram = sorted(int(c) for c in np.unique(kh))\n'),
    ('        dls = {c: vo_han(lo(TapAnh(thu_muc, tr.image[tr.center == c], tr.target[tr.center == c], True),\n',
     '        dls = {c: vo_han(lo(TapAnh(thu_muc, tr.image[kh == c], tr.target[kh == c], True),\n'),
    ('        n_k = {c: int((tr.center == c).sum()) for c in tram}\n',
     '        n_k = {c: int((kh == c).sum()) for c in tram}\n'),
]
for cu, moi in thay:
    if s.count(cu) != 1:
        sys.exit(f"LỖI: không tìm thấy đúng 1 chỗ:\n{cu}")
    s = s.replace(cu, moi)
ast.parse(s); p.write_text(s, encoding="utf-8"); print("fed_isic.py: đã vá (--kich-ban nguon|tron).")
