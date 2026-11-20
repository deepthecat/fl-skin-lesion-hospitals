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
