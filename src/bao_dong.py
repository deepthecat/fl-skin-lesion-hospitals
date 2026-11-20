"""bao_dong.py — gửi một thông báo wandb (email / app) khi hàng đợi chạy đêm kết thúc.

    python bao_dong.py "Đêm 9 xong" "12/12 lượt, xem dem9.txt"

Đặt ở cuối mọi script hàng đợi. Thông báo cho run bị LỖI giữa chừng thì bật trong wandb.ai -> Settings -> Alerts
("Run crashed"). Chưa login hoặc lỗi mạng thì chỉ in ra màn hình, không làm hỏng gì.
"""
import sys

from nhat_ky import DU_AN, GOC, co_khoa

tieu_de = sys.argv[1] if len(sys.argv) > 1 else "ham-fl"
noi_dung = sys.argv[2] if len(sys.argv) > 2 else ""
print(f"[báo động] {tieu_de}: {noi_dung}")
try:
    import wandb
    if co_khoa(wandb):
        r = wandb.init(project=DU_AN, name=f"thong_bao: {tieu_de}", job_type="thong_bao", dir=str(GOC), reinit=True)
        r.alert(title=tieu_de, text=noi_dung or tieu_de)
        r.finish()
except Exception as e:
    print(f"(bao_dong) không gửi được: {e}")
