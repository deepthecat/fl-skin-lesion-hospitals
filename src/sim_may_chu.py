"""
sim_may_chu.py — bước 4 thân cây chính: máy chủ "tò mò" thấy gì, và GỘP BẢO MẬT che được bao nhiêu. Cần GPU (ngắn).

    python sim_may_chu.py

Mô phỏng MỘT vòng FedAvg ở vòng tốt nhất của các lượt FedAvg nguồn thật đã chạy (ConvNeXt-atto, EfficientNet-B0, 3 seed):
  1. Nạp mô hình chung tốt nhất (ketqua/lien_ket/<tên>.pt) và tốc độ học ở đúng vòng đó (lấy từ lich_su).
  2. Mỗi bệnh viện học tại chỗ 1 epoch y hệt huan_luyen_lien_ket.py -> 4 mô hình riêng = thứ máy chủ thấy KHI KHÔNG có gộp bảo mật.
  3. Gộp FedAvg 4 mô hình riêng -> thứ máy chủ thấy KHI CÓ gộp bảo mật (chỉ thấy tổng/trung bình).
Tấn công từng bệnh viện k: thành viên = ảnh train CỦA k, không thành viên = ảnh val CỦA CHÍNH k (bản 2: bản 1 dùng val của
mọi bệnh viện nên bị lẫn độ khó giữa các nguồn — AUC < 0,5 ở rosendahl/dias). Chấm bằng loss và RMIA
(8 mô hình tham chiếu đêm 15). So: mô hình riêng của k  vs  mô hình đã gộp  vs  mô hình chung trước vòng đó.
Không cài mật mã thật: gộp bảo mật chỉ đổi THỨ máy chủ nhìn thấy, không đổi việc huấn luyện, nên mô phỏng là đủ cho câu hỏi riêng tư.
"""
import json
import types
import warnings

import numpy as np
import pandas as pd
import torch

from du_lieu import RA, doc_chia_tap, nap_dem, tao_mo_hinh
from huan_luyen_tap_trung import suy_luan, trong_so_lop
from huan_luyen_lien_ket import huan_luyen_may_tram
from rmia import chi_so, diem_rmia, p_dung
from phan_tich_dp_sau import nap_tham_chieu

warnings.filterwarnings("ignore")
TEN_MAY = {0: "molemax", 1: "modern", 2: "rosendahl", 3: "dias"}


def chay_mot(bb, s, ten, fpt, v_tot, kieu_lr, lr, ds, X, y, may, nguon_cua, w_lop, arg, tb, hang):
    phan = ds.phan.values
    torch.manual_seed(s)
    mh = tao_mo_hinh(bb, pretrained=False).to(tb)
    w0 = {k: v.cpu() for k, v in torch.load(fpt, map_location="cpu").items()}
    mh.load_state_dict(w0)
    rng = np.random.default_rng(100 + s)
    scaler = torch.amp.GradScaler(enabled=tb.type == "cuda")
    rieng, n_k = {}, {}
    for k, idx in may.items():
        rieng[k], _ = huan_luyen_may_tram(mh, w0, X, y, idx, arg, lr, w_lop, tb, rng, scaler)
        n_k[k] = len(idx)
    tong = sum(n_k.values())
    gop = {}
    for kk, v in w0.items():
        if torch.is_floating_point(v):
            gop[kk] = sum(rieng[k][kk].float() * (n_k[k] / tong) for k in may).to(v.dtype)
        else:
            gop[kk] = rieng[0][kk]
    p_cua = {}
    for nhan, w in [("chung_truoc", w0), ("da_gop", gop)] + [(f"rieng_{k}", rieng[k]) for k in may]:
        mh.load_state_dict(w)
        p_cua[nhan] = p_dung(suy_luan(mh, X, np.arange(len(ds)), tb), y)
    for k, idx in may.items():
        val_k = (phan == "val") & (ds.nguon.values == nguon_cua[k])
        chon = np.zeros(len(ds), bool); chon[idx] = True; chon |= val_k
        la_tv = (np.isin(np.arange(len(ds)), idx))[chon].astype(int)
        ref = nap_tham_chieu(bb, ds, chon, la_tv)
        for nhan in ("chung_truoc", "da_gop", f"rieng_{k}"):
            p = p_cua[nhan][chon]
            r = dict(mo_hinh=bb, seed=s, benh_vien=TEN_MAY.get(k, k), n=len(idx), n_val=int(val_k.sum()), kieu_lr=kieu_lr, vong=v_tot + 1, lr=lr,
                     may_chu_thay={"chung_truoc": "mô hình chung trước vòng", "da_gop": "CÓ gộp bảo mật (chỉ thấy trung bình)",
                                   f"rieng_{k}": "KHÔNG gộp bảo mật (thấy mô hình riêng)"}[nhan])
            r["AUC_loss"], r["TPR1_loss"] = chi_so(np.log(p + 1e-12), la_tv)
            if ref is not None:
                P, T, a = ref
                r["AUC_rmia"], r["TPR1_rmia"] = chi_so(diem_rmia(p, P, T, a), la_tv)
            hang.append(r)


def main():
    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = doc_chia_tap()
    X = nap_dem(ds, 224, "p1")
    y = ds.nhan.values
    phan = ds.phan.values
    tr = np.flatnonzero(phan == "train")
    m = ds["may_nguon"].values
    may = {int(k): np.flatnonzero((phan == "train") & (m == k)) for k in sorted(np.unique(m[tr]))}
    w_lop = trong_so_lop(y[tr], "nghich").to(tb)
    arg = types.SimpleNamespace(epoch_cuc_bo=1, bs=64, thuat_toan="fedavg", mu=0.01)
    nguon_cua = {k: ds.nguon.values[idx[0]] for k, idx in may.items()}
    hang = []
    for bb in ("convnext_atto", "efficientnet_b0"):
        for s in range(3):
            ten = f"fedavg_nguon_{bb}_txlp1_kt224_s{s}"
            fpt, fj = RA / "lien_ket" / f"{ten}.pt", RA / "lien_ket" / f"{ten}.json"
            if not (fpt.exists() and fj.exists()):
                print(f"thiếu {ten}.pt/.json — bỏ qua"); continue
            ls = json.load(open(fj))["lich_su"]
            v_tot = int(np.argmax([r["val_bacc"] for r in ls]))
            for kieu_lr, lr in (("lr vòng tốt nhất", float(ls[v_tot]["lr"])), ("lr vòng 10", float(ls[9]["lr"]))):
                chay_mot(bb, s, ten, fpt, v_tot, kieu_lr, lr, ds, X, y, may, nguon_cua, w_lop, arg, tb, hang)
            print(f"xong {ten} (vòng tốt nhất {v_tot + 1})", flush=True)
    b = pd.DataFrame(hang)
    b.to_csv(RA / "sim_may_chu_v2.csv", index=False)
    cot = [c for c in ("AUC_loss", "AUC_rmia", "TPR1_rmia") if c in b.columns]
    pd.set_option("display.width", 220)
    print("\nMức lộ của từng bệnh viện theo thứ máy chủ nhìn thấy (TB 3 seed):\n")
    for kl in b.kieu_lr.unique():
        x = b[b.kieu_lr == kl]
        print(f"--- máy chủ quan sát ở {kl} (lr TB {x.lr.mean():.1e})")
        print(x.groupby(["mo_hinh", "benh_vien", "may_chu_thay"], sort=False)[cot + ["n_val"]].mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nĐọc: chênh 'KHÔNG gộp' − 'CÓ gộp' = phần mà gộp bảo mật che được trước máy chủ tò mò. 'CÓ gộp' vẫn lộ bằng"
          "\nmô hình cuối — phần đó chỉ DP mới che. Thành viên = ảnh train của bệnh viện, không thành viên = ảnh val CỦA CHÍNH bệnh viện đó (n_val nhỏ -> số dao động).")
    print(f"Đã ghi {RA / 'sim_may_chu_v2.csv'}")


if __name__ == "__main__":
    main()
