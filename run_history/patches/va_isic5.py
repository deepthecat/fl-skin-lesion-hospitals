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
