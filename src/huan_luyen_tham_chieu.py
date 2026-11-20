"""
huan_luyen_tham_chieu.py — mô hình THAM CHIẾU cho tấn công RMIA (bước 2 của thân cây chính).

    python huan_luyen_tham_chieu.py --backbone convnext_atto --cap 0 --nua 0

Ý tưởng: kẻ tấn công tự huấn luyện vài mô hình "giống mô hình nạn nhân" trên một NỬA dữ liệu cùng phân phối.
Với mỗi ảnh x, các mô hình tham chiếu KHÔNG học x cho biết "một mô hình bình thường tự tin cỡ nào với x".
Ảnh khó (lớp hiếm, ảnh mờ) thì mọi mô hình đều kém tự tin — nên loss cao không còn bị nhầm thành "không phải thành viên".

Cách chia: tập train chia đôi THEO TỔN THƯƠNG (ảnh cùng một nốt luôn ở cùng một nửa).
  --cap j (0..3) chọn một cách chia ngẫu nhiên; --nua 0/1 chọn nửa nào. Hai nửa của cùng một cặp bù nhau,
  nên với 4 cặp × 2 nửa = 8 mô hình, mỗi ảnh train nằm NGOÀI đúng 4 mô hình. Ảnh val nằm ngoài cả 8.
Huấn luyện y hệt huan_luyen_tap_trung.py (AdamW 3e-4, OneCycle, lô 64, trọng số lớp nghịch đảo, chọn epoch tốt nhất
theo val), chỉ khác là học trên một nửa train và KHÔNG ghi wandb (đây là công cụ của kẻ tấn công, không phải thí nghiệm).
Ghi ra ketqua/tham_chieu/tc_*.npz: logit của MỌI ảnh + mặt nạ 'trong' (ảnh nào mô hình này đã học).
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from du_lieu import RA, TIEN_XU_LY, cham, doc_chia_tap, ma_file, nap_dem, tao_mo_hinh
from huan_luyen_tap_trung import lo, suy_luan, trong_so_lop


def mat_na_nua(ds, cap, nua):
    """Ảnh train thuộc nửa `nua` của cách chia `cap` (chia theo tổn thương)."""
    tr = ds.phan.values == "train"
    les = np.array(sorted(ds.lesion_id[tr].unique()))
    rng = np.random.default_rng(1000 + cap)
    chon = set(les[rng.permutation(len(les))[: len(les) // 2]])
    trong = tr & ds.lesion_id.isin(chon).values
    return trong if nua == 0 else (tr & ~trong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--cap", type=int, required=True, help="cách chia 0..3")
    ap.add_argument("--nua", type=int, required=True, choices=(0, 1))
    ap.add_argument("--kt", type=int, default=224)
    ap.add_argument("--epoch", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tien-xu-ly", default="p1", choices=tuple(TIEN_XU_LY))
    ap.add_argument("--khong-pretrained", action="store_true", help="chỉ để thử code")
    ap.add_argument("--ra", default=str(RA / "tham_chieu"))
    arg = ap.parse_args()

    ten = f"tc_{arg.backbone}_txl{arg.tien_xu_ly}_kt{arg.kt}_c{arg.cap}n{arg.nua}"
    thu_muc = Path(arg.ra); thu_muc.mkdir(parents=True, exist_ok=True)
    f_npz = thu_muc / f"{ten}.npz"
    if f_npz.exists():
        print(f"Đã có {f_npz.name} — bỏ qua."); return
    s = 100 + 2 * arg.cap + arg.nua
    torch.manual_seed(s); np.random.seed(s)
    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    X = nap_dem(ds, arg.kt, arg.tien_xu_ly)
    y = ds.nhan.values
    trong = mat_na_nua(ds, arg.cap, arg.nua)
    I_tr, I_val = np.flatnonzero(trong), np.flatnonzero(ds.phan.values == "val")
    print(f"{ten}: học {len(I_tr)} ảnh train ({trong.sum() / (ds.phan.values == 'train').sum():.0%}), chọn epoch theo {len(I_val)} ảnh val | {tb}")

    mh = tao_mo_hinh(arg.backbone, pretrained=not arg.khong_pretrained).to(tb)
    w = trong_so_lop(y[I_tr], "nghich").to(tb)
    tong_buoc = arg.epoch * math.ceil(len(I_tr) / arg.bs)
    opt = torch.optim.AdamW(mh.parameters(), lr=arg.lr, weight_decay=1e-4)
    lich = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=arg.lr, total_steps=tong_buoc, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=tb.type == "cuda")
    rng = np.random.default_rng(s)
    tot, w_tot, t0 = -1.0, None, time.time()
    for ep in range(arg.epoch):
        mh.train()
        p = rng.permutation(I_tr)
        for i in range(0, len(p), arg.bs):
            idx = p[i:i + arg.bs]
            if len(idx) >= 2:                                   # BatchNorm không chạy với lô 1 ảnh
                xb, yb = lo(X, idx, tb, tang=True), torch.from_numpy(y[idx]).to(tb)
                with torch.autocast(tb.type, enabled=tb.type == "cuda"):
                    loss = F.cross_entropy(mh(xb), yb, weight=w)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            lich.step()
        v = cham(y[I_val], suy_luan(mh, X, I_val, tb))["bacc"]
        if v > tot:
            tot, w_tot = v, copy.deepcopy({k: t.detach().cpu() for k, t in mh.state_dict().items()})
        print(f"  epoch {ep + 1:>2}/{arg.epoch}  val bacc {v:.4f}  ({time.time() - t0:.0f}s){' *' if v == tot else ''}", flush=True)

    mh.load_state_dict(w_tot)
    logit = suy_luan(mh, X, np.arange(len(ds)), tb)
    np.savez_compressed(f_npz, image=np.asarray(ds.image.tolist(), dtype=np.str_),
                        phan=np.asarray(ds.phan.tolist(), dtype=np.str_), nhan=y,
                        logit=logit.astype(np.float32), trong=trong)
    json.dump(dict(ten=ten, ma_script=ma_file(__file__), cau_hinh=vars(arg), val_bacc=tot, giay=time.time() - t0,
                   so_anh_hoc=int(len(I_tr))), open(thu_muc / f"{ten}.json", "w"), indent=1, ensure_ascii=False)
    print(f"Xong {ten}: val bacc tốt nhất {tot:.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
