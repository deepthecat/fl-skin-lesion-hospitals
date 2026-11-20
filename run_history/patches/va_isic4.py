"""va_isic4.py — thêm số vòng / số bước vào tên lượt fed_isic.py khi khác mặc định (để lượt 27×33 không bị bỏ qua vì 'đã có'). Chạy lại vô hại."""
from pathlib import Path
f = Path("fed_isic.py"); s = f.read_text()
cu = 'ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{ctt}{kbt}_s{arg.seed}"'
moi = ('vbt = f"_v{arg.vong}b{arg.so_buoc}" if (arg.vong or arg.so_buoc != 100) else ""\n'
       '    ten = f"isic_{arg.che_do}{hau}_{arg.anh}{bbt}{opt_t}{ctt}{kbt}{vbt}_s{arg.seed}"')
if "vbt =" in s:
    print("va_isic4: đã vá từ trước.")
elif cu in s:
    f.write_text(s.replace(cu, moi)); print("va_isic4: đã vá tên (thêm _v{vòng}b{bước}).")
else:
    raise SystemExit("va_isic4: KHÔNG tìm thấy dòng tên — dán lỗi này cho Claude.")
