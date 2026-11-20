"""
nhat_ky.py — ghi nhật ký thí nghiệm lên Weights & Biases (wandb), dùng chung cho mọi script huấn luyện.

Ý tưởng: mỗi lượt chạy = một "run" trên wandb, tên trùng tên file kết quả (ten). Ghi:
  - cấu hình (mọi tham số dòng lệnh) + mã băm của code + commit git hiện tại (và code có bị sửa chưa commit không),
  - đường cong từng epoch/vòng (val bAcc, loss, độ trôi, cosine...),
  - kết quả cuối (val/test) và file .json kết quả,
  - bản sao các file code đã sinh ra kết quả đó.
Nhóm (group) = tên bỏ hậu tố seed, nên 3 seed của cùng cấu hình nằm chung một nhóm trên wandb.

An toàn khi chạy qua đêm:
  - chưa cài wandb, hoặc đặt HAMFL_WANDB=0  -> không làm gì, script vẫn chạy bình thường;
  - chưa `wandb login`                      -> tự chuyển chế độ offline (không bao giờ dừng lại hỏi mật khẩu);
    sau đó đẩy lên bằng: wandb sync ~/ham_fl/wandb/offline-run-*
  - wandb lỗi giữa chừng                    -> in cảnh báo, việc huấn luyện KHÔNG bị ảnh hưởng.

Gắn giả thuyết (đặt trong script hàng đợi, trước lệnh python):
  export HAMFL_GIA_THUYET="G8b,A1"           -> thành tag trên wandb, lọc theo giả thuyết
  export HAMFL_DU_DOAN="LN thu hẹp phần máy"  -> ghi vào config TRƯỚC khi chạy (bằng chứng đã đoán trước)
Dữ liệu chia tập (ketqua/chia_tap.csv) được ghi thành Artifact "chia_tap"; nội dung không đổi thì cùng một phiên bản,
nên mỗi run trỏ đúng phiên bản chia nó đã dùng.
"""
import hashlib
import os
import re
import subprocess
from pathlib import Path

DU_AN = os.environ.get("HAMFL_WANDB_PROJECT", "ham-fl")
GOC = Path(__file__).parent


def git_hien_tai():
    try:
        c = subprocess.run(["git", "-C", str(GOC), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        b = subprocess.run(["git", "-C", str(GOC), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=5)
        if c.returncode != 0:
            return None, None
        return c.stdout.strip(), bool(b.stdout.strip())
    except Exception:
        return None, None


def phang(d, tien_to=""):
    """Làm phẳng dict có list/list-lồng để wandb vẽ được: do_troi=[a,b] -> do_troi/0, do_troi/1; cos=[[..]] -> cos/0_1."""
    ra = {}
    for k, v in d.items():
        k = f"{tien_to}{k}"
        if isinstance(v, dict):
            ra.update(phang(v, f"{k}/"))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, (list, tuple)):
                    for j, z in enumerate(x):
                        if i < j and isinstance(z, (int, float)):
                            ra[f"{k}/{i}_{j}"] = z
                elif isinstance(x, (int, float)):
                    ra[f"{k}/{i}"] = x
        elif isinstance(v, (int, float, bool)):
            ra[k] = v
    return ra


def co_khoa(wandb):
    """Đã `wandb login` chưa — kiểm tra mà KHÔNG BAO GIỜ dừng lại hỏi (an toàn khi chạy qua đêm)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return bool(wandb.login(prompt=False, verify=False))
    except TypeError:
        try:
            return bool(wandb.api.api_key)
        except Exception:
            return False
    except Exception:
        return False


def nhom_cua(ten):
    return re.sub(r"_s\d+$", "", ten)


class NhatKy:
    def __init__(self, ten, cau_hinh, loai, file_code=(), them=None):
        self.run = None
        if os.environ.get("HAMFL_WANDB", "1") == "0":
            return
        try:
            import wandb
        except ImportError:
            print("(nhat_ky) chưa cài wandb — bỏ qua ghi nhật ký. Cài: pip install wandb")
            return
        try:
            che_do = os.environ.get("WANDB_MODE")
            if che_do is None and not co_khoa(wandb):
                che_do = "offline"
                print("(nhat_ky) chưa wandb login -> ghi OFFLINE vào ~/ham_fl/wandb, đẩy lên sau bằng `wandb sync`")
            commit, ban_nhap = git_hien_tai()
            gt = [g.strip() for g in os.environ.get("HAMFL_GIA_THUYET", "").split(",") if g.strip()]
            try:
                from du_lieu import RA
                chia = RA / "chia_tap.csv"
            except Exception:
                chia = GOC / "ketqua" / "chia_tap.csv"
            ma_chia = hashlib.sha256(chia.read_bytes()).hexdigest()[:10] if chia.exists() else None
            cfg = dict(cau_hinh, loai=loai, git_commit=commit, git_chua_commit=ban_nhap, ma_chia_tap=ma_chia,
                       gia_thuyet=",".join(gt), du_doan=os.environ.get("HAMFL_DU_DOAN", ""), **(them or {}))
            self.run = wandb.init(project=DU_AN, name=ten, id=re.sub(r"[^A-Za-z0-9_.-]", "-", ten)[:120],
                                  group=nhom_cua(ten), job_type=loai, config=cfg, resume="allow",
                                  dir=str(GOC), mode=che_do or "online", reinit=True,
                                  tags=[loai] + gt + ([] if commit else ["khong_git"]))
            for f in file_code:
                if Path(f).exists():
                    self.run.save(str(f), base_path=str(GOC), policy="now")
        except Exception as e:
            print(f"(nhat_ky) wandb lỗi khi khởi tạo, bỏ qua: {e}")
            self.run = None
            return
        try:
            if chia.exists():
                a = wandb.Artifact("chia_tap", type="dataset", description="phép chia theo tổn thương + cột kịch bản client")
                a.add_file(str(chia))
                self.run.use_artifact(a)
        except Exception as e:
            print(f"(nhat_ky) không ghi được artifact chia_tap (không ảnh hưởng huấn luyện): {e}")

    def ghi(self, d, buoc):
        if self.run is None:
            return
        try:
            self.run.log(phang(d), step=int(buoc))
        except Exception as e:
            print(f"(nhat_ky) lỗi ghi: {e}")

    def ket_thuc(self, ket_qua, file_json=None, them=None):
        if self.run is None:
            return
        try:
            for k, v in phang(dict(ket_qua=ket_qua, **(them or {}))).items():
                self.run.summary[k] = v
            if file_json and Path(file_json).exists():
                self.run.save(str(file_json), base_path=str(Path(file_json).parent.parent), policy="now")
            self.run.finish()
        except Exception as e:
            print(f"(nhat_ky) lỗi khi kết thúc: {e}")
