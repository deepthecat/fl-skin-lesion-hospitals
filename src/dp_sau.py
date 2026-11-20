"""
dp_sau.py — bước 3 thân cây chính: DP-SGD SÂU HƠN lớp cuối, trong học liên kết thật (FedAvg, 4 bệnh viện).

    python dp_sau.py --backbone convnext_atto --muc-mo 1 --epsilon 3 --seed 0

Ý tưởng. Mô hình học trước trên ImageNet (dữ liệu công khai). Phần đầu mạng ĐÓNG BĂNG; chỉ phần đuôi được học:
  --muc-mo 1 (L1): khối cuối của xương sống + đầu phân loại
  --muc-mo 2 (L2): hai khối cuối + đầu phân loại
  ConvNeXt-atto : L1 = stages[3] (+ norm_pre, head) ≈ 1,9 triệu tham số; L2 = stages[2:] ≈ 3,3 triệu
  EfficientNet-B0: L1 = blocks[6] + conv_head/bn2 + classifier ≈ 1,0 triệu; L2 = blocks[5:] ≈ 2,9 triệu
Toàn mô hình ở chế độ suy luận khi học (model.eval()): LayerNorm vốn không phụ thuộc lô; BatchNorm thì KHOÁ sổ tay
ImageNet (dùng thống kê cố định) — nên gradient từng ảnh tính được, DP-SGD áp được cho cả hai họ mô hình.
Đó là phép so ở CỔNG A: LayerNorm tự chỉnh theo từng ảnh vs BatchNorm khoá sổ ImageNet, cùng mức ε.

DP-SGD tại mỗi bệnh viện (giống dp_tuyen_tinh.py, nhưng gradient từng ảnh tính bằng torch.func.vmap):
  mỗi bước lấy mẫu Poisson (mỗi ảnh vào lô với xác suất q = bs/n_k), cắt gradient từng ảnh về độ dài ≤ C,
  cộng lại, thêm nhiễu Gauss N(0, (σC)²), chia cho kích thước lô kỳ vọng, rồi cập nhật bằng Adam (hậu xử lý, không tốn ε).
  σ của từng bệnh viện tính bằng kế toán RDP (Opacus) cho đúng số bước của nó qua mọi vòng: mỗi bệnh viện tiêu ngân sách ε riêng.
Mức: inf (không cắt, không nhiễu) · clip (chỉ cắt) · 8 · 3 · 1.
Máy chủ: FedAvg theo số ảnh. Chọn vòng tốt nhất theo val (val không phải dữ liệu huấn luyện, không tốn ε — ghi trong bài).
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
from torch.func import functional_call, grad, vmap

try:   # PyTorch 2.14 tự chuyển phép bmm trên CUDA sang nhân Triton (cần biên dịch C + Python.h lúc chạy). Tắt đi -> dùng cuBLAS như cũ.
    from torch._native import registry as _native_registry
    _native_registry.deregister_op_overrides(disable_dsl_names="triton")
except Exception:
    pass

from nhat_ky import NhatKy
from du_lieu import RA, TIEN_XU_LY, cham, doc_chia_tap, ma_file, nap_dem, tao_mo_hinh
from huan_luyen_tap_trung import lo, trong_so_lop
from dp_tuyen_tinh import sigma_cho


class Duoi(torch.nn.Module):
    """Phần đuôi được học. dau(x) chạy phần đầu đóng băng (không cần gradient)."""

    def __init__(self, mh, ten, k):
        super().__init__()
        self.ten, self.k = ten, k
        if ten.startswith("convnext"):
            self.stages = torch.nn.Sequential(*list(mh.stages)[4 - k:])
            self.norm_pre, self.head = mh.norm_pre, mh.head
            self._dau = [mh.stem] + list(mh.stages)[:4 - k]
        elif ten.startswith("efficientnet"):
            nb = len(mh.blocks)
            self.stages = torch.nn.Sequential(*list(mh.blocks)[nb - k:])
            self.conv_head, self.bn2, self.global_pool, self.classifier = mh.conv_head, mh.bn2, mh.global_pool, mh.classifier
            self._dau = [mh.conv_stem, mh.bn1] + list(mh.blocks)[:nb - k]
        else:
            raise SystemExit(f"Chưa định nghĩa phần đuôi cho {ten}")

    @torch.no_grad()
    def dau(self, x):
        for m in self._dau:
            x = m(x)
        return x

    def forward(self, x):
        x = self.stages(x)
        if self.ten.startswith("convnext"):
            return self.head(self.norm_pre(x))
        return self.classifier(self.global_pool(self.bn2(self.conv_head(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--muc-mo", type=int, default=1, choices=(1, 2))
    ap.add_argument("--epsilon", default="3", help="inf, clip, hoặc một số")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kich-ban", default="nguon")
    ap.add_argument("--tien-xu-ly", default="p1", choices=tuple(TIEN_XU_LY))
    ap.add_argument("--kt", type=int, default=224)
    ap.add_argument("--vong", type=int, default=30)
    ap.add_argument("--epoch-cuc-bo", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256, help="lô kỳ vọng (Poisson); DP cần lô lớn")
    ap.add_argument("--vi-lo", type=int, default=32, help="lô nhỏ đưa vào GPU khi tính gradient từng ảnh")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--khong-pretrained", action="store_true", help="chỉ để thử code")
    ap.add_argument("--dung-sau-vong", type=int, default=0, help="chỉ để thử code / đo tốc độ")
    ap.add_argument("--ra", default=str(RA / "dp_sau"))
    arg = ap.parse_args()

    txl = "" if arg.tien_xu_ly == "p0" else f"_txl{arg.tien_xu_ly}"
    tag = (f"_b{arg.bs}" if arg.bs != 256 else "") + (f"_lr{arg.lr:g}" if arg.lr != 1e-3 else "")   # mặc định giữ tên cũ
    ten = f"dpsau_{arg.kich_ban}_L{arg.muc_mo}_eps{arg.epsilon}_{arg.backbone}{txl}{tag}_kt{arg.kt}_s{arg.seed}"
    thu_muc = Path(arg.ra); thu_muc.mkdir(parents=True, exist_ok=True)
    f_json = thu_muc / f"{ten}.json"
    if f_json.exists() and not arg.dung_sau_vong:
        print(f"Đã có {f_json.name} — bỏ qua."); return
    torch.manual_seed(arg.seed)
    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    X = nap_dem(ds, arg.kt, arg.tien_xu_ly)
    y = ds.nhan.values
    I = {p: np.flatnonzero(ds.phan.values == p) for p in ("train", "val", "val_chinh_thuc", "test")}
    if "trung_train" in ds.columns:
        I["test_sach"] = np.flatnonzero((ds.phan.values == "test") & (ds.trung_train.values == 0))
    m = ds[f"may_{arg.kich_ban}"].values
    may = {int(k): np.flatnonzero((ds.phan.values == "train") & (m == k)) for k in sorted(np.unique(m[I["train"]]))}

    mh = tao_mo_hinh(arg.backbone, pretrained=not arg.khong_pretrained).to(tb).eval()
    for p in mh.parameters():
        p.requires_grad_(False)
    duoi = Duoi(mh, arg.backbone, arg.muc_mo).to(tb).eval()
    ten_p = [n for n, _ in duoi.named_parameters()]
    so_mo = sum(p.numel() for p in duoi.parameters())
    bufs = {n: b for n, b in duoi.named_buffers()}
    w_lop = trong_so_lop(y[I["train"]], "nghich").to(tb)

    cat, nhieu = arg.epsilon != "inf", arg.epsilon not in ("inf", "clip")
    tt = {}
    for k, idx in may.items():
        q = min(1.0, arg.bs / len(idx))
        bv = math.ceil(1 / q) * arg.epoch_cuc_bo
        s, e = sigma_cho(arg.epsilon, arg.delta, q, bv * arg.vong) if nhieu else (0.0, math.inf)
        tt[k] = dict(n=len(idx), q=q, buoc_moi_vong=bv, sigma=s, eps_that=e)
    print(f"{ten} | {so_mo:,} tham số được học | {tb}")
    for k, t in tt.items():
        print(f"   bệnh viện {k}: {t['n']} ảnh, q={t['q']:.3f}, {t['buoc_moi_vong']} bước/vòng, σ={t['sigma']:.3f}, ε thật={t['eps_that']:.2f}")

    def mat_mat(p, b, x, yy):
        out = functional_call(duoi, (p, b), (x.unsqueeze(0),))
        lp = F.log_softmax(out.float(), dim=1)[0]                 # viết tay: cross_entropy kiểm nhãn bằng .item(), vmap không cho
        t = yy.unsqueeze(0)
        return -(w_lop.gather(0, t) * lp.gather(0, t)).sum()

    g_tung_anh = vmap(grad(mat_mat), in_dims=(None, None, 0, 0))

    def tong_grad(p, idx, tang):
        """Tổng gradient (đã cắt nếu cần) của các ảnh idx, dict tên -> tensor."""
        tong = {n: torch.zeros_like(v) for n, v in p.items()}
        for i in range(0, len(idx), arg.vi_lo):
            b = idx[i:i + arg.vi_lo]
            with torch.autocast(tb.type, enabled=tb.type == "cuda"):
                h = duoi.dau(lo(X, b, tb, tang=tang))
            h = h.float()
            yb = torch.from_numpy(y[b]).to(tb)
            G = g_tung_anh(p, bufs, h, yb)
            if cat:
                do_dai = torch.sqrt(sum(G[n].flatten(1).pow(2).sum(1) for n in ten_p))
                he_so = torch.clamp(arg.clip / (do_dai + 1e-12), max=1.0)
                for n in ten_p:
                    G[n] = G[n] * he_so.view(-1, *([1] * (G[n].dim() - 1)))
            for n in ten_p:
                tong[n] += G[n].sum(0)
        return tong

    @torch.no_grad()
    def suy_luan_full(p, idx):
        ra = []
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            with torch.autocast(tb.type, enabled=tb.type == "cuda"):
                ra.append(functional_call(duoi, (p, bufs), (duoi.dau(lo(X, b, tb)),)).float().cpu())
        return torch.cat(ra).numpy()

    w_tc = {n: v.detach().clone() for n, v in duoi.named_parameters()}
    rng = np.random.default_rng(arg.seed)
    gen = torch.Generator(device=tb).manual_seed(arg.seed)
    tot, w_tot, lich_su, t0 = -1.0, None, [], time.time()
    nk = NhatKy(ten, vars(arg), "dp_sau", [__file__], them=dict(so_tham_so_mo=so_mo,
                sigma=str([round(t["sigma"], 3) for t in tt.values()])))
    for vong in range(arg.vong):
        lr = arg.lr * 0.5 * (1 + math.cos(math.pi * vong / arg.vong))
        cac, n_k = [], []
        for k, idx in may.items():
            t = tt[k]
            p = {n: v.clone().requires_grad_(False) for n, v in w_tc.items()}
            opt_p = [torch.nn.Parameter(v) for v in p.values()]
            opt = torch.optim.Adam(opt_p, lr=lr)
            B = max(t["q"] * t["n"], 1.0)
            for _ in range(t["buoc_moi_vong"]):
                chon = idx[rng.random(len(idx)) < t["q"]]
                cur = {n: v.detach() for n, v in zip(p.keys(), opt_p)}
                G = tong_grad(cur, chon, tang=True) if len(chon) else {n: torch.zeros_like(v) for n, v in cur.items()}
                for (n, v), par in zip(G.items(), opt_p):
                    if nhieu:
                        v = v + torch.randn(v.shape, generator=gen, device=tb) * t["sigma"] * arg.clip
                    par.grad = v / B
                opt.step(); opt.zero_grad(set_to_none=True)
            cac.append({n: v.detach() for n, v in zip(p.keys(), opt_p)}); n_k.append(t["n"])
        tong = sum(n_k)
        w_tc = {n: sum(c[n] * (nn_ / tong) for c, nn_ in zip(cac, n_k)) for n in ten_p}
        v = cham(y[I["val"]], suy_luan_full(w_tc, I["val"]))
        dau = ""
        if v["bacc"] > tot:
            tot, w_tot, dau = v["bacc"], {n: x.clone() for n, x in w_tc.items()}, " *"
        lich_su.append(dict(vong=vong + 1, lr=lr, **{f"val_{a}": b for a, b in v.items()}))
        nk.ghi(lich_su[-1], vong + 1)
        print(f"vòng {vong + 1:>3}/{arg.vong}  val bacc {v['bacc']:.4f}  ({time.time() - t0:.0f}s){dau}", flush=True)
        if arg.dung_sau_vong and vong + 1 >= arg.dung_sau_vong:
            print(f"(dừng thử) ~{(time.time() - t0) / (vong + 1):.0f} s/vòng -> ~{(time.time() - t0) / (vong + 1) * arg.vong / 60:.0f} phút/lượt")
            return

    logit = suy_luan_full(w_tot, np.arange(len(ds)))
    ket = {p: cham(y[i], logit[i]) for p, i in I.items() if len(i)}
    np.savez_compressed(thu_muc / f"{ten}_logit.npz", image=np.asarray(ds.image.tolist(), dtype=np.str_),
                        phan=np.asarray(ds.phan.tolist(), dtype=np.str_), nhan=y, logit=logit.astype(np.float32))
    json.dump(dict(ten=ten, ma_script=ma_file(__file__), ma_du_lieu=ma_file(Path(__file__).parent / "du_lieu.py"),
                   epsilon=arg.epsilon, cau_hinh=vars(arg), so_tham_so_mo=so_mo, giay=time.time() - t0,
                   may_tram={str(k): v for k, v in tt.items()}, ket_qua=ket, lich_su=lich_su),
              open(f_json, "w"), indent=1, ensure_ascii=False, default=str)
    nk.ket_thuc(ket, f_json, dict(giay=time.time() - t0))
    print(f"Xong {ten}: val {ket['val']['bacc']:.4f}  test {ket['test']['bacc']:.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
