"""
kiem_tra_wandb.py — đối chiếu 3 nguồn cho từng con số chính của paper:
  (1) số ghi trong paper, (2) file JSON kết quả trên máy (ketqua/*/*.json), (3) các run trên wandb (project ham-fl).
Ngoài ra so từng run wandb với file JSON cùng tên (khớp / lệch / thiếu).

Chạy trong ~/ham_fl, sau khi đã `wandb login` từ trước (script KHÔNG đọc, KHÔNG in API key):
    python kiem_tra_wandb.py                    # đủ 3 nguồn (cần mạng)
    python kiem_tra_wandb.py --chi-local        # chỉ paper vs JSON, không cần mạng
    python kiem_tra_wandb.py --ketqua DUONG_DAN # dùng thư mục ketqua khác
Ra: bảng trên màn hình + ketqua/kiem_tra_wandb.csv (từng run wandb: trạng thái so với JSON).
"""
import argparse, csv, json, os, re
from pathlib import Path
from statistics import mean

DU_AN = os.environ.get("HAMFL_WANDB_PROJECT", "ham-fl")
SAI_SO_RUN = 1e-6     # run wandb và JSON phải giống hệt (cùng một con số được ghi hai nơi)
SAI_SO_PAPER = 6e-4   # paper làm tròn 3 chữ số

HAM = [("EfficientNet-B0", "efficientnet_b0", 224), ("ResNet-18", "resnet18", 224),
       ("ResNet-50 BN", "resnet50.a1h_in1k", 160), ("ResNet-50 GN", "resnet50_gn.a1h_in1k", 160),
       ("ConvNeXt-atto", "convnext_atto", 224), ("DeiT-tiny", "deit_tiny_patch16_224", 224)]
P_TT = [0.778, 0.797, 0.800, 0.791, 0.813, 0.779]   # Bảng III: tập trung
P_NG = [0.659, 0.657, 0.698, 0.700, 0.737, 0.742]   # Bảng III: FedAvg site thật
P_TR = [0.707, 0.706, 0.748, 0.713, 0.733, 0.698]   # Bảng III: FedAvg site trộn


def kiem_tra_paper():
    """(nhãn, regex tên run, khoá kết quả, số trong paper, gom theo lr rồi lấy lr tốt nhất?)"""
    C = []
    for (nhan, bb, kt), a, b, c in zip(HAM, P_TT, P_NG, P_TR):
        e = re.escape(bb)
        C.append((f"Bảng III tập trung {nhan}", rf"{e}_kt{kt}_nghich_txlp1_s[0-2]", "ket_qua/val/bacc", a, False))
        C.append((f"Bảng III FedAvg thật {nhan}", rf"fedavg_nguon_{e}_txlp1_kt{kt}_s[0-2]", "ket_qua/val/bacc", b, False))
        C.append((f"Bảng III FedAvg trộn {nhan}", rf"fedavg_tron_nguon_{e}_txlp1_kt{kt}_s[0-2]", "ket_qua/val/bacc", c, False))
    C += [("E: ConvNeXt FedAvg 5 seed (val)", r"fedavg_nguon_convnext_atto_txlp1_kt224_s[0-4]", "ket_qua/val/bacc", 0.741, False),
          ("E: ConvNeXt FedAvg 5 seed (test)", r"fedavg_nguon_convnext_atto_txlp1_kt224_s[0-4]", "ket_qua/test/bacc", 0.662, False),
          ("EfficientNet FedAvg 5 seed (val)", r"fedavg_nguon_efficientnet_b0_txlp1_kt224_s[0-4]", "ket_qua/val/bacc", 0.661, False),
          ("EfficientNet FedAvg 5 seed (test)", r"fedavg_nguon_efficientnet_b0_txlp1_kt224_s[0-4]", "ket_qua/test/bacc", 0.597, False)]
    for e, a, d in (("inf", 0.617, 0.710), ("8", 0.506, 0.628), ("3", 0.430, 0.554), ("1", 0.325, 0.455)):
        C.append((f"Bảng V ImageNet eps={e}", rf"dptt_nguon_eps{e}_efficientnet_b0_txlp1_kt224_s[0-2]", "ket_qua/val/bacc", a, True))
        C.append((f"Bảng V BCN eps={e}", rf"dptt_nguon_eps{e}_efficientnet_b0_bcn_txlp1_kt224_s[0-2]", "ket_qua/val/bacc", d, True))
    for nhan, mau, tb, gop in (("ISIC tập trung 60 ep EfficientNet", "isic_tap_trung_cc_efficientnet_b0_ham_e60", 0.777, 0.815),
                               ("ISIC tập trung 60 ep ConvNeXt", "isic_tap_trung_cc_convnext_atto_ham_e60", 0.736, 0.791),
                               ("ISIC FedAvg 27x100 EfficientNet", "isic_fedavg_cc_efficientnet_b0_ham_v27b100", 0.640, 0.699),
                               ("ISIC FedAvg 27x100 ConvNeXt", "isic_fedavg_cc_convnext_atto_ham_v27b100", 0.683, 0.700),
                               ("ISIC trộn 27x100 EfficientNet", "isic_fedavg_cc_efficientnet_b0_ham_tron_v27b100", 0.697, 0.747),
                               ("ISIC trộn 27x100 ConvNeXt", "isic_fedavg_cc_convnext_atto_ham_tron_v27b100", 0.691, 0.730),
                               ("ISIC FedAvg 9x100 EfficientNet", "isic_fedavg_cc_efficientnet_b0_ham", 0.643, None),
                               ("ISIC FedAvg 9x100 ConvNeXt", "isic_fedavg_cc_convnext_atto_ham", 0.628, None),
                               ("ISIC FedAvg 27x33 EfficientNet", "isic_fedavg_cc_efficientnet_b0_ham_v27b33", 0.638, None),
                               ("ISIC FedAvg 27x33 ConvNeXt", "isic_fedavg_cc_convnext_atto_ham_v27b33", 0.639, None),
                               ("ISIC FLamby tập trung", "isic_tap_trung_cc", 0.731, None),
                               ("ISIC FLamby FedAvg", "isic_fedavg_cc", 0.624, None),
                               ("ISIC FLamby FedProx", "isic_fedprox_cc", 0.652, None)):
        C.append((nhan + " (theo trung tâm)", rf"{mau}_s[0-2]", "ket_qua/test_tb_tram", tb, False))
        if gop is not None:
            C.append((nhan + " (ảnh gộp)", rf"{mau}_s[0-2]", "ket_qua/test_gop", gop, False))
    return C


def phang(d, tt=""):
    ra = {}
    for k, v in d.items():
        k2 = f"{tt}{k}"
        if isinstance(v, dict):
            ra.update(phang(v, k2 + "/"))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            ra[k2] = float(v)
    return ra


def doc_local(ra):
    L = []
    for f in sorted(ra.glob("*/*.json")):
        if f.name.startswith("."):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "ket_qua" not in d:
            continue
        L.append(dict(ten=d.get("ten", f.stem), nguon=f"{f.parent.name}/{f.name}",
                      kq=phang(d["ket_qua"], "ket_qua/"), lr=(d.get("cau_hinh") or {}).get("lr")))
    return L


def doc_wandb():
    import wandb
    api = wandb.Api(timeout=90)
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    duong = f"{entity}/{DU_AN}"
    W = []
    for r in api.runs(duong, per_page=500):
        s = {}
        for k, v in r.summary._json_dict.items():
            if k.startswith("ket_qua/") and isinstance(v, (int, float)) and not isinstance(v, bool):
                s[k] = float(v)
        W.append(dict(ten=r.name, nguon=r.url, kq=s, lr=r.config.get("lr"), trang_thai=r.state))
    return duong, W


def gia_tri(ds, mau, khoa, theo_lr):
    """Trung bình trên các run khớp mẫu; mỗi tên chỉ tính một lần. Nếu theo_lr: tính riêng từng lr, lấy lr cao nhất."""
    nhom = {}
    for x in ds:
        ten = re.sub(r"_lr[0-9.]+$", "", x["ten"])          # run DP trên wandb có thêm đuôi _lr16, _lr4, _lr1
        if re.fullmatch(mau, ten) and khoa in x["kq"]:
            g = x["lr"] if theo_lr else None
            nhom.setdefault(g, {}).setdefault(ten, x["kq"][khoa])
    if not nhom:
        return None, 0, None
    tot = max(nhom, key=lambda g: mean(nhom[g].values()))
    return mean(nhom[tot].values()), len(nhom[tot]), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi-local", action="store_true")
    ap.add_argument("--ketqua", default=None)
    arg = ap.parse_args()
    if arg.ketqua:
        ra = Path(arg.ketqua).expanduser()
    else:
        try:
            from du_lieu import RA as ra
        except Exception:
            ra = Path.home() / "ham_fl" / "ketqua"
    L = doc_local(ra)
    print(f"JSON trên máy: {len(L)} file có kết quả ({ra})")
    W, duong = [], None
    if not arg.chi_local:
        duong, W = doc_wandb()
        print(f"wandb: {len(W)} run trong {duong}  ->  https://wandb.ai/{duong}")

    # ---- Phần 1: từng con số của paper
    print("\nPHẦN 1 — số trong paper | JSON trên máy | wandb")
    loi = 0
    for nhan, mau, khoa, p, theo_lr in kiem_tra_paper():
        j, nj, lj = gia_tri(L, mau, khoa, theo_lr)
        dong = f"  {nhan:<46} paper {p:.3f} | JSON " + (f"{j:.3f} (n={nj}{', lr=' + str(lj) if theo_lr else ''})" if j is not None else "KHÔNG THẤY")
        ok = j is not None and abs(j - p) <= SAI_SO_PAPER
        if W:
            w, nw, lw = gia_tri(W, mau, khoa, theo_lr)
            dong += " | wandb " + (f"{w:.3f} (n={nw})" if w is not None else "KHÔNG THẤY")
            ok = ok and w is not None and abs(w - j) <= SAI_SO_RUN * 10
        loi += not ok
        print(("   OK " if ok else " >>XEM") + dong[1:])
    print(f"  => {loi} dòng cần xem lại")

    if not W:
        return
    # ---- Phần 2: từng run wandb so với JSON cùng tên
    theo_ten = {}
    for x in L:
        theo_ten.setdefault(x["ten"], []).append(x)
    dem, hang = {}, []
    ten_w = set()
    for r in W:
        ten_w.add(re.sub(r"_lr[0-9.]+$", "", r["ten"]))
        if not r["kq"]:
            tt, ghi = "khong_co_ket_qua", r["trang_thai"]
        elif re.sub(r"_lr[0-9.]+$", "", r["ten"]) not in theo_ten:
            tt, ghi = "chi_co_tren_wandb", ""
        else:
            lech = []
            for x in theo_ten[re.sub(r"_lr[0-9.]+$", "", r["ten"])]:
                chung = set(r["kq"]) & set(x["kq"])
                d = max((abs(r["kq"][k] - x["kq"][k]) for k in chung), default=None)
                lech.append((d, x["nguon"]))
            tot = min((t for t in lech if t[0] is not None), default=(None, ""))
            tt = "khop" if tot[0] is not None and tot[0] <= SAI_SO_RUN else "lech"
            ghi = f"lệch tối đa {tot[0]:.4g} so với {tot[1]}" if tt == "lech" and tot[0] is not None else tot[1]
        dem[tt] = dem.get(tt, 0) + 1
        hang.append(dict(ten=r["ten"], trang_thai=tt, ghi_chu=ghi, url=r["nguon"]))
    for x in L:
        if x["ten"] not in ten_w:
            dem["chi_co_tren_may"] = dem.get("chi_co_tren_may", 0) + 1
            hang.append(dict(ten=x["ten"], trang_thai="chi_co_tren_may", ghi_chu=x["nguon"], url=""))
    print("\nPHẦN 2 — từng run wandb so với JSON cùng tên:")
    for k in ("khop", "lech", "chi_co_tren_wandb", "chi_co_tren_may", "khong_co_ket_qua"):
        print(f"  {k:<20}{dem.get(k, 0):>5}")
    for h in [h for h in hang if h["trang_thai"] == "lech"][:30]:
        print(f"   lệch: {h['ten']}  ({h['ghi_chu']})")
    f = ra / "kiem_tra_wandb.csv"
    with open(f, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=["ten", "trang_thai", "ghi_chu", "url"])
        w.writeheader()
        w.writerows(hang)
    print(f"\nChi tiết từng run: {f}")


if __name__ == "__main__":
    main()
