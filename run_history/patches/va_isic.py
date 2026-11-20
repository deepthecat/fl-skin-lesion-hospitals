"""va_isic.py — thêm vào fed_isic.py: chọn xương sống timm (--backbone) và tối ưu Adam cho FL (--toi-uu-fl adam).
Mặc định giữ nguyên giao thức FLamby và tên file cũ. Chạy nhiều lần vẫn an toàn."""
import ast
import pathlib
import sys

p = pathlib.Path("fed_isic.py"); s = p.read_text(encoding="utf-8")
if "--toi-uu-fl" in s:
    print("fed_isic.py: đã vá từ trước."); sys.exit(0)
thay = [
    ('def tao_mo_hinh(pretrained=True):\n    from efficientnet_pytorch import EfficientNet\n',
     'def tao_mo_hinh(pretrained=True, backbone="effnet_flamby"):\n'
     '    if backbone != "effnet_flamby":                 # xương sống timm (cùng thư viện cho mọi mô hình khi so với nhau)\n'
     '        import timm\n'
     '        return timm.create_model(backbone, pretrained=pretrained, num_classes=8)\n'
     '    from efficientnet_pytorch import EfficientNet\n'),
    ('    ap.add_argument("--ra", default=str(RA / "fed_isic"))\n',
     '    ap.add_argument("--ra", default=str(RA / "fed_isic"))\n'
     '    ap.add_argument("--backbone", default="effnet_flamby", help="effnet_flamby (gói efficientnet_pytorch như FLamby) hoặc tên timm")\n'
     '    ap.add_argument("--toi-uu-fl", default="sgd", choices=("sgd", "adam"), help="FL tại chỗ: sgd 0,01 (FLamby) hoặc adam 5e-4")\n'),
    ('    ten = f"isic_{arg.che_do}{hau}_{arg.anh}_s{arg.seed}"\n',
     '    bbt = "" if arg.backbone == "effnet_flamby" else f"_{arg.backbone}"\n'
     '    opt_t = "_adam" if (arg.toi_uu_fl == "adam" and arg.che_do not in ("tap_trung", "cuc_bo")) else ""\n'
     '    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}_s{arg.seed}"\n'),
    ('    mh = tao_mo_hinh(not arg.khong_pretrained).to(tb)\n',
     '    mh = tao_mo_hinh(not arg.khong_pretrained, arg.backbone).to(tb)\n'),
    ('                opt = torch.optim.SGD(mh.parameters(), lr=0.01)\n',
     '                opt = (torch.optim.SGD(mh.parameters(), lr=0.01) if arg.toi_uu_fl == "sgd"\n'
     '                       else torch.optim.Adam(mh.parameters(), lr=5e-4))\n'),
]
for cu, moi in thay:
    if s.count(cu) != 1:
        sys.exit(f"LỖI: không tìm thấy đúng 1 chỗ:\n{cu}")
    s = s.replace(cu, moi)
ast.parse(s); p.write_text(s, encoding="utf-8"); print("fed_isic.py: đã vá (--backbone, --toi-uu-fl).")
