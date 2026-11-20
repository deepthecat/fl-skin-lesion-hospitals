"""va_isic3.py — thêm --cong-thuc ham cho fed_isic.py: dùng ĐÚNG công thức huấn luyện của HAM10000 cho cả hai mô hình
(tập trung: AdamW 3e-4, weight decay 1e-4, khởi động 10% rồi giảm theo cos; FL: AdamW 3e-4 giảm theo cos qua các vòng).
Lý do: đêm 20 ConvNeXt học không nổi với Adam 5e-4 không có lịch → so kiến trúc bị lẫn chuyện optimizer.
Mặc định (flamby) giữ nguyên giao thức và tên file cũ. Chạy nhiều lần vẫn an toàn."""
import ast
import pathlib
import sys

p = pathlib.Path("fed_isic.py"); s = p.read_text(encoding="utf-8")
if "--cong-thuc" in s:
    print("fed_isic.py: đã vá công thức từ trước."); sys.exit(0)
if "--kich-ban" not in s:
    sys.exit("LỖI: chạy va_isic.py và va_isic2.py trước")
thay = [
    ('    ap.add_argument("--kich-ban",',
     '    ap.add_argument("--cong-thuc", default="flamby", choices=("flamby", "ham"),\n'
     '                    help="flamby: Adam 5e-4 (tập trung), SGD/Adam theo --toi-uu-fl (FL) | ham: AdamW 3e-4 + lịch cos như HAM10000")\n'
     '    ap.add_argument("--kich-ban",'),
    ('    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{kbt}_s{arg.seed}"\n',
     '    ctt = "_ham" if arg.cong_thuc == "ham" else ""\n'
     '    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{ctt}{kbt}_s{arg.seed}"\n'),
    ('        opt = torch.optim.Adam(mh.parameters(), lr=5e-4)\n        ep0 = 0\n',
     '        opt = (torch.optim.Adam(mh.parameters(), lr=5e-4) if arg.cong_thuc == "flamby"\n'
     '               else torch.optim.AdamW(mh.parameters(), lr=3e-4, weight_decay=1e-4))\n        ep0 = 0\n'),
    ('            print(f"Chạy tiếp từ epoch {ep0 + 1}")\n        for ep in range(ep0, arg.epoch):\n',
     '            print(f"Chạy tiếp từ epoch {ep0 + 1}")\n'
     '        lich = None\n'
     '        if arg.cong_thuc == "ham":     # khởi động 10% rồi giảm theo cos (giống OneCycle của HAM10000); không cần lưu trạng thái\n'
     '            nb, tong = len(dl), arg.epoch * len(dl)\n'
     '            def he_so(t, nb0=ep0 * nb):\n'
     '                t += nb0; k = int(0.1 * tong)\n'
     '                return max(0.04, t / max(k, 1)) if t < k else 0.5 * (1 + math.cos(math.pi * (t - k) / max(tong - k, 1)))\n'
     '            lich = torch.optim.lr_scheduler.LambdaLR(opt, he_so)\n'
     '        def mot(x, y):\n'
     '            r = buoc(mh, opt, scaler, crit, x, y, tb)\n'
     '            if lich is not None:\n'
     '                lich.step()\n'
     '            return r\n'
     '        for ep in range(ep0, arg.epoch):\n'),
    ('            l = [buoc(mh, opt, scaler, crit, x, y, tb) for x, y in dl]\n',
     '            l = [mot(x, y) for x, y in dl]\n'),
    ('                opt = (torch.optim.SGD(mh.parameters(), lr=0.01) if arg.toi_uu_fl == "sgd"\n'
     '                       else torch.optim.Adam(mh.parameters(), lr=5e-4))\n',
     '                if arg.cong_thuc == "ham":\n'
     '                    opt = torch.optim.AdamW(mh.parameters(), lr=3e-4 * 0.5 * (1 + math.cos(math.pi * v / vong)), weight_decay=1e-4)\n'
     '                else:\n'
     '                    opt = (torch.optim.SGD(mh.parameters(), lr=0.01) if arg.toi_uu_fl == "sgd"\n'
     '                           else torch.optim.Adam(mh.parameters(), lr=5e-4))\n'),
]
for cu, moi in thay:
    if s.count(cu) != 1:
        sys.exit(f"LỖI: không tìm thấy đúng 1 chỗ:\n{cu}")
    s = s.replace(cu, moi)
ast.parse(s); p.write_text(s, encoding="utf-8"); print("fed_isic.py: đã vá (--cong-thuc ham).")
