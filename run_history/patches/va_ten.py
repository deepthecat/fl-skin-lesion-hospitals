"""va_ten.py — vá tên run để các biến thể ablation KHÔNG trùng tên (file và wandb) với run gốc. Chạy nhiều lần vẫn an toàn."""
import ast
import pathlib
import sys


def va(file, cu, moi, dau):
    p = pathlib.Path(file); s = p.read_text(encoding="utf-8")
    if dau in s:
        print(f"{file}: đã vá từ trước."); return
    if s.count(cu) != 1:
        sys.exit(f"LỖI: {file} — không tìm thấy đúng 1 chỗ cần vá")
    s = s.replace(cu, moi); ast.parse(s); p.write_text(s, encoding="utf-8"); print(f"{file}: đã vá.")


# 1) FL: thêm hậu tố khi tắt học trước / đổi trọng số lớp / đổi số epoch tại chỗ (mặc định giữ tên cũ)
va("huan_luyen_lien_ket.py",
   '    ten = f"{arg.thuat_toan}{mu}_{arg.kich_ban}_{arg.backbone}{db}{txl}{cb}{gh}{vg}_kt{arg.kt}_s{arg.seed}"',
   '    hau_ab = ("_khp" if arg.khong_pretrained else "") + ("" if arg.trong_so == "nghich" else f"_ts{arg.trong_so}") \\\n'
   '        + ("" if arg.epoch_cuc_bo == 1 else f"_e{arg.epoch_cuc_bo}")      # ablation: tên riêng, không đè run gốc\n'
   '    ten = f"{arg.thuat_toan}{mu}_{arg.kich_ban}_{arg.backbone}{db}{txl}{cb}{gh}{vg}{hau_ab}_kt{arg.kt}_s{arg.seed}"',
   "{hau_ab}_kt")
# 2) DP lớp cuối: số vòng khác 30 -> hậu tố _v{vong} trong tên file; lr cố định -> hậu tố trong tên wandb
va("dp_tuyen_tinh.py",
   '        ten = f"dptt_{kich_ban}_eps{muc}_{arg.backbone}{txl}_kt{arg.kt}_s{seed}"',
   '        vg = f"_v{arg.vong}" if arg.vong != 30 else ""\n'
   '        ten = f"dptt_{kich_ban}_eps{muc}_{arg.backbone}{txl}{vg}_kt{arg.kt}_s{seed}"\n'
   '        ten_nk = ten + (f"_lr{lr:g}" if arg.lr > 0 else "")   # wandb: lr cố định có id riêng (thư mục khác vẫn cùng tên file)',
   "ten_nk = ten")
va("dp_tuyen_tinh.py", "        nk = NhatKy(ten, ", "        nk = NhatKy(ten_nk, ", "NhatKy(ten_nk, ")
