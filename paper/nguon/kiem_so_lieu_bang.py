"""Đối chiếu từng ô số trong Bảng III, V, VI và các hình 4–6 với JSON kết quả."""
import json, re, glob
from statistics import mean, stdev
from pathlib import Path

R = Path("ketqua")
LOI = []


def doc(mau, khoa="val/bacc", thu_muc="*"):
    out = {}
    for f in glob.glob(str(R / thu_muc / "*.json")):
        d = json.load(open(f))
        ten = d.get("ten", Path(f).stem)
        if re.fullmatch(mau, ten):
            v = d["ket_qua"]
            for k in khoa.split("/"):
                v = v[k]
            out[ten] = v
    return out


def ss(nhan, paper, gt, tol=6e-4):
    ok = abs(paper - gt) <= tol
    if not ok:
        LOI.append((nhan, paper, gt))
    print(f"{'OK ' if ok else '>>X'} {nhan:<55} paper {paper:+.3f}  json {gt:+.4f}")


# ---------- Bảng III
HAM = [("EfficientNet-B0", "efficientnet_b0", 224, .778, .659, .707, 40),
       ("ResNet-18", "resnet18", 224, .797, .657, .706, 35),
       ("ResNet-50 BN", "resnet50.a1h_in1k", 160, .800, .698, .748, 49),
       ("ResNet-50 GN", "resnet50_gn.a1h_in1k", 160, .791, .700, .713, 15),
       ("ConvNeXt-atto", "convnext_atto", 224, .813, .737, .733, None),
       ("DeiT-tiny", "deit_tiny_patch16_224", 224, .779, .742, .698, None)]
print("== Bảng III")
T3 = {}
for nhan, bb, kt, a, b, c, sh in HAM:
    e = re.escape(bb)
    tt = doc(rf"{e}_kt{kt}_nghich_txlp1_s[0-2]")
    ng = doc(rf"fedavg_nguon_{e}_txlp1_kt{kt}_s[0-2]")
    tr = doc(rf"fedavg_tron_nguon_{e}_txlp1_kt{kt}_s[0-2]")
    assert len(tt) == len(ng) == len(tr) == 3, (nhan, len(tt), len(ng), len(tr))
    A, B, C = mean(tt.values()), mean(ng.values()), mean(tr.values())
    T3[nhan] = (A, B, C)
    ss(f"{nhan} central", a, A); ss(f"{nhan} source", b, B); ss(f"{nhan} mixed", c, C)
    ss(f"{nhan} gap", round(a - b, 3), A - B)
    share = (C - B) / (A - B) * 100
    print(f"     share json {share:.1f}%  paper {sh}")
    if sh is not None and abs(share - sh) > 0.6:
        LOI.append((f"{nhan} share", sh, share))
    # dấu theo seed
    d = [tr[k.replace('fedavg_nguon', 'fedavg_tron_nguon')] - ng[k] for k in sorted(ng)]
    print(f"     mixed−source theo seed: {[round(x, 3) for x in d]}")
for e in ("deit_tiny_patch16_224",):
    for n in (1, 2):
        v = doc(rf"fedavg_tron_nguon{n}_{e}_txlp1_kt224_s[0-2]")
        print(f"  DeiT draw {n}: {mean(v.values()):.4f} (n={len(v)})")

print("== 5 seed")
for bb, pv, pt in (("convnext_atto", .741, .662), ("efficientnet_b0", .661, .597)):
    v = doc(rf"fedavg_nguon_{bb}_txlp1_kt224_s[0-4]")
    t = doc(rf"fedavg_nguon_{bb}_txlp1_kt224_s[0-4]", "test/bacc")
    assert len(v) == 5
    ss(f"{bb} 5 seed val", pv, mean(v.values())); ss(f"{bb} 5 seed test", pt, mean(t.values()))
    print(f"     sd val {stdev(v.values()):.4f}  min {min(v.values()):.3f} max {max(v.values()):.3f}")

# ---------- Bảng V (lr tốt nhất theo val, cho mỗi eps)
print("== Bảng V")
P5 = {"": {"inf": (.617, .565), "8": (.506, .436), "3": (.430, .386), "1": (.325, .297)},
      "_bcn": {"inf": (.710, .652), "8": (.628, .567), "3": (.554, .508), "1": (.455, .403)}}
for duoi, cot in P5.items():
    thu = ["dp_tuyen_tinh_lr4", "dp_tuyen_tinh_lr1", "dp_tuyen_tinh"] if duoi == "" else \
          ["dp_tuyen_tinh_bcn_lr16", "dp_tuyen_tinh_bcn_lr4", "dp_tuyen_tinh_bcn_lr1"]
    for eps, (pv, pt) in cot.items():
        best = None
        for td in thu:
            mau = rf"dptt_nguon_eps{eps}_efficientnet_b0{duoi}_txlp1_kt224_s[0-2]"
            v = doc(mau, "val/bacc", td); t = doc(mau, "test/bacc", td)
            if len(v) == 3:
                m = mean(v.values())
                lr = json.load(open(glob.glob(str(R / td / f"dptt_nguon_eps{eps}_efficientnet_b0{duoi}_txlp1_kt224_s0.json"))[0]))["cau_hinh"]["lr"]
                print(f"     {duoi or 'imagenet'} eps{eps} {td} lr={lr}: val {m:.4f} test {mean(t.values()):.4f}")
                if best is None or m > best[0]:
                    best = (m, mean(t.values()), td)
        ss(f"V {duoi or 'ImageNet'} eps{eps} val", pv, best[0]); ss(f"V {duoi or 'ImageNet'} eps{eps} test", pt, best[1])

# ---------- Bảng VI + ISIC
print("== Bảng VI / Fed-ISIC")
for nhan, mau, tb, gop in (("9x100 Eff", "isic_fedavg_cc_efficientnet_b0_ham", .643, None),
                           ("9x100 Conv", "isic_fedavg_cc_convnext_atto_ham", .628, None),
                           ("27x33 Eff", "isic_fedavg_cc_efficientnet_b0_ham_v27b33", .638, None),
                           ("27x33 Conv", "isic_fedavg_cc_convnext_atto_ham_v27b33", .639, None),
                           ("27x100 Eff", "isic_fedavg_cc_efficientnet_b0_ham_v27b100", .640, .699),
                           ("27x100 Conv", "isic_fedavg_cc_convnext_atto_ham_v27b100", .683, .700),
                           ("mix 27x100 Eff", "isic_fedavg_cc_efficientnet_b0_ham_tron_v27b100", .697, .747),
                           ("mix 27x100 Conv", "isic_fedavg_cc_convnext_atto_ham_tron_v27b100", .691, .730),
                           ("central60 Eff", "isic_tap_trung_cc_efficientnet_b0_ham_e60", .777, .815),
                           ("central60 Conv", "isic_tap_trung_cc_convnext_atto_ham_e60", .736, .791),
                           ("FLamby central", "isic_tap_trung_cc", .731, None),
                           ("FLamby FedAvg", "isic_fedavg_cc", .624, None),
                           ("FLamby FedProx", "isic_fedprox_cc", .652, None)):
    v = doc(rf"{mau}_s[0-2]", "test_tb_tram")
    assert len(v) == 3, (mau, len(v))
    ss(f"ISIC {nhan} per-centre", tb, mean(v.values()))
    if gop:
        ss(f"ISIC {nhan} pooled", gop, mean(doc(rf"{mau}_s[0-2]", "test_gop").values()))
    if "27x100" in nhan and "mix" not in nhan:
        print("     seeds:", [round(x, 3) for x in v.values()])

print("\nLỖI:", LOI if LOI else "không có")
