"""
huan_luyen_tap_trung.py — baseline TẬP TRUNG: gom hết dữ liệu train về một chỗ rồi huấn luyện.

Đây là trần mà federated cố tiến gần (RQ1), và là mô hình đem so với bảng xếp hạng ISIC 2018.

    # đo tốc độ trước (35 bước, không huấn luyện thật):
    python huan_luyen_tap_trung.py --backbone efficientnet_b0 --kt 224 --do-toc
    # chạy thật:
    python huan_luyen_tap_trung.py --backbone efficientnet_b0 --kt 224 --epoch 15 --seed 0
    # bước 5 — đổi kiểu tiền xử lý (p0 ép vuông / p1 giữ tỉ lệ / p2 giữ tỉ lệ + chuẩn hoá màu):
    python huan_luyen_tap_trung.py --backbone efficientnet_b0 --kt 224 --seed 0 --tien-xu-ly p2

Quy trình:
  - Chọn mô hình tốt nhất theo balanced accuracy trên VALIDATION (tách theo tổn thương).
  - Tập TEST chính thức chỉ chấm MỘT lần, bằng mô hình đã chọn, ở cuối. Không dùng để chọn gì cả.
  - Lưu logit của MỌI ảnh (train/val/test). Bước tấn công suy luận thành viên (RQ2) dùng lại
    đúng các logit này, khỏi phải huấn luyện lại.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nhat_ky import NhatKy
from du_lieu import DL, LOP, RA, TB, TIEN_XU_LY, cham, doc_chia_tap, ma_file, nap_dem, tao_mo_hinh


def tang_cuong(x):
    """x: float [B,3,H,W] trong [0,1]. Ảnh soi da không có chiều 'lên trên' -> lật và xoay 90° đều hợp lệ."""
    B = x.shape[0]
    m = torch.rand(B, device=x.device) < 0.5
    x[m] = x[m].flip(3)
    m = torch.rand(B, device=x.device) < 0.5
    x[m] = x[m].flip(2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))
    a = 1 + 0.4 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)     # độ tương phản ±20%
    b = 0.2 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)         # độ sáng ±0.1
    return (x * a + b).clamp(0, 1)


def lo(X, idx, tb, tang=False):
    x = torch.from_numpy(X[idx]).to(tb, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    if tang:
        x = tang_cuong(x)
    tb_, dl_ = torch.tensor(TB, device=tb).view(1, 3, 1, 1), torch.tensor(DL, device=tb).view(1, 3, 1, 1)
    return (x - tb_) / dl_


@torch.no_grad()
def suy_luan(mh, X, idx, tb, bs=128):
    mh.eval()
    ra = []
    for i in range(0, len(idx), bs):
        with torch.autocast(tb.type, enabled=tb.type == "cuda"):
            ra.append(mh(lo(X, idx[i:i + bs], tb)).float().cpu())
    return torch.cat(ra).numpy()


def trong_so_lop(y, kieu):
    if kieu == "khong":
        return None
    n = np.bincount(y, minlength=len(LOP)).astype(np.float64)
    w = n.sum() / (len(LOP) * np.maximum(n, 1))      # nghịch đảo tần suất
    if kieu == "can":
        w = np.sqrt(w)                                # dịu hơn, đỡ dao động với lớp cực hiếm
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0",
                    help="tên mô hình trong timm: resnet18, efficientnet_b0, mobilenetv3_large_100 ...")
    ap.add_argument("--kt", type=int, default=224, help="cạnh ảnh")
    ap.add_argument("--epoch", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trong-so", default="nghich", choices=("nghich", "can", "khong"),
                    help="trọng số lớp trong loss: nghịch đảo tần suất / căn bậc hai của nó / không")
    ap.add_argument("--tien-xu-ly", default="p0", choices=tuple(TIEN_XU_LY),
                    help="; ".join(f"{k}: {v}" for k, v in TIEN_XU_LY.items()))
    ap.add_argument("--do-toc", action="store_true", help="chỉ đo tốc độ và VRAM rồi thoát")
    ap.add_argument("--khong-pretrained", action="store_true", help="chỉ để thử code")
    ap.add_argument("--ra", default=str(RA / "tap_trung"))
    arg = ap.parse_args()

    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    X = nap_dem(ds, arg.kt, arg.tien_xu_ly)
    y = ds.nhan.values
    I = {p: np.flatnonzero(ds.phan.values == p) for p in ("train", "val", "val_chinh_thuc", "test")}
    if "trung_train" in ds.columns:       # do kiem_trung_test.py ghi vào: bỏ ảnh test trùng ảnh train
        I["test_sach"] = np.flatnonzero((ds.phan.values == "test") & (ds.trung_train.values == 0))
    print(" | ".join(f"{p}: {len(v)}" for p, v in I.items()), "|", tb)

    mh = tao_mo_hinh(arg.backbone, pretrained=not arg.khong_pretrained).to(tb)
    so_tham_so = sum(p.numel() for p in mh.parameters())
    w = trong_so_lop(y[I["train"]], arg.trong_so)
    w = w.to(tb) if w is not None else None
    buoc_moi_epoch = math.ceil(len(I["train"]) / arg.bs)
    tong_buoc = arg.epoch * buoc_moi_epoch
    opt = torch.optim.AdamW(mh.parameters(), lr=arg.lr, weight_decay=1e-4)
    lich = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=arg.lr, total_steps=tong_buoc, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=tb.type == "cuda")

    def mot_buoc(idx):
        xb = lo(X, idx, tb, tang=True)
        yb = torch.from_numpy(y[idx]).to(tb)
        with torch.autocast(tb.type, enabled=tb.type == "cuda"):
            loss = F.cross_entropy(mh(xb), yb, weight=w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        lich.step()
        return float(loss.detach())

    # ---------------- đo tốc độ
    if arg.do_toc:
        mh.train()
        rng = np.random.default_rng(0)
        for _ in range(5):
            mot_buoc(rng.choice(I["train"], arg.bs))
        if tb.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0, n = time.time(), 30
        for _ in range(n):
            mot_buoc(rng.choice(I["train"], arg.bs))
        if tb.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        anh_s = n * arg.bs / dt
        vram = torch.cuda.max_memory_allocated() / 2**30 if tb.type == "cuda" else float("nan")
        print(f"{arg.backbone:<24} kt={arg.kt:<4} {so_tham_so/1e6:5.1f}M tham số | {anh_s:6.0f} ảnh/s | "
              f"VRAM đỉnh {vram:.2f} GB | ~{len(I['train'])/anh_s:.0f} s/epoch")
        return

    # ---------------- huấn luyện
    txl = "" if arg.tien_xu_ly == "p0" else f"_txl{arg.tien_xu_ly}"     # p0 giữ tên cũ
    ten = f"{arg.backbone}_kt{arg.kt}_{arg.trong_so}{txl}_s{arg.seed}"
    thu_muc = Path(arg.ra); thu_muc.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(arg.seed)
    tot, lich_su, t0 = -1.0, [], time.time()
    nk = NhatKy(ten, vars(arg), "tap_trung", [__file__, Path(__file__).parent / "du_lieu.py"],
                them=dict(so_tham_so=so_tham_so, ma_script=ma_file(__file__)))
    for ep in range(arg.epoch):
        mh.train()
        p = rng.permutation(I["train"])
        l = [mot_buoc(p[i:i + arg.bs]) for i in range(0, len(p), arg.bs)]
        v = cham(y[I["val"]], suy_luan(mh, X, I["val"], tb))
        lich_su.append(dict(epoch=ep + 1, loss=float(np.mean(l)), **{f"val_{k}": x for k, x in v.items()}))
        nk.ghi(lich_su[-1], ep + 1)
        dau = ""
        if v["bacc"] > tot:
            tot, dau = v["bacc"], " *"
            torch.save(mh.state_dict(), thu_muc / f"{ten}.pt")
        print(f"epoch {ep+1:>2}/{arg.epoch}  loss {np.mean(l):.4f}  val bacc {v['bacc']:.4f}  "
              f"f1 {v['f1_macro']:.4f}  acc {v['acc']:.4f}  ({time.time()-t0:.0f}s){dau}", flush=True)

    # ---------------- chấm cuối bằng mô hình tốt nhất trên val
    mh.load_state_dict(torch.load(thu_muc / f"{ten}.pt", map_location=tb))
    tat_ca = np.arange(len(ds))
    logit = suy_luan(mh, X, tat_ca, tb)
    ket = {p: cham(y[i], logit[i]) for p, i in I.items() if len(i)}
    np.savez_compressed(thu_muc / f"{ten}_logit.npz", image=np.asarray(ds.image.tolist(), dtype=np.str_),
                        phan=np.asarray(ds.phan.tolist(), dtype=np.str_), nhan=y, logit=logit.astype(np.float32))
    json.dump(dict(ten=ten, ma_script=ma_file(__file__), ma_du_lieu=ma_file(Path(__file__).parent / "du_lieu.py"),
                   cau_hinh=vars(arg), so_tham_so=so_tham_so, giay=time.time() - t0,
                   ket_qua=ket, lich_su=lich_su),
              open(thu_muc / f"{ten}.json", "w"), indent=1, ensure_ascii=False)
    nk.ket_thuc(ket, thu_muc / f"{ten}.json", dict(giay=time.time() - t0))
    print("\nKẾT QUẢ (mô hình tốt nhất trên val):")
    for p, d in ket.items():
        print(f"  {p:<15} bacc {d['bacc']:.4f}  f1 {d['f1_macro']:.4f}  acc {d['acc']:.4f}  auc {d['auc_macro']:.4f}")
    print(f"Đã ghi {thu_muc / (ten + '.json')}")


if __name__ == "__main__":
    main()
