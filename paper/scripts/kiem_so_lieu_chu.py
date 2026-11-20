"""Số trong chữ (IV-B, IV-C, IV-E, IV-F) đối chiếu JSON."""
import json, glob, re
from statistics import mean
from pathlib import Path
R = Path("ketqua")


def doc(mau, khoa="val/bacc", thu_muc="*"):
    out = {}
    for f in glob.glob(str(R / thu_muc / "*.json")):
        d = json.load(open(f))
        ten = d.get("ten", Path(f).stem)
        if re.fullmatch(mau, ten):
            v = d
            for k in ("ket_qua/" + khoa).split("/"):
                v = v[k]
            out[ten] = v
    return out


def m(mau, khoa="val/bacc", thu_muc="*"):
    d = doc(mau, khoa, thu_muc)
    return (round(mean(d.values()), 4), len(d)) if d else (None, 0)


def theo_seed(m1, m2, thu_muc="*"):
    a, b = doc(m1, thu_muc=thu_muc), doc(m2, thu_muc=thu_muc)
    ka, kb = sorted(a), sorted(b)
    return [round(b[y] - a[x], 3) for x, y in zip(ka, kb)]


S = r"_s[0-2]"
print("--- IV-B FedBN")
for bb in ("efficientnet_b0", "resnet18"):
    gh = "_gh" if bb == "efficientnet_b0" else ""
    print(bb, "fedavg src", m(rf"fedavg_nguon_{bb}_txlp1_kt224{S}"), "fedbn src", m(rf"fedbn_nguon_{bb}_txlp1{gh}_kt224{S}"),
          "fedavg mix", m(rf"fedavg_tron_nguon_{bb}_txlp1_kt224{S}"), "fedbn mix", m(rf"fedbn_tron_nguon_{bb}_txlp1{gh}_kt224{S}"))
    print("   fedbn test (ensemble)", m(rf"fedbn_nguon_{bb}_txlp1{gh}_kt224{S}", "test/bacc"), "fedavg test", m(rf"fedavg_nguon_{bb}_txlp1_kt224{S}", "test/bacc"))
    print("   fedbn val_ens", m(rf"fedbn_nguon_{bb}_txlp1{gh}_kt224{S}", "val_ens/bacc"))
print("--- P2 colour constancy")
for bb in ("efficientnet_b0", "convnext_atto"):
    print(bb, "central p1", m(rf"{bb}_kt224_nghich_txlp1{S}"), "p2", m(rf"{bb}_kt224_nghich_txlp2{S}"),
          "| fedavg p1", m(rf"fedavg_nguon_{bb}_txlp1_kt224{S}"), "p2", m(rf"fedavg_nguon_{bb}_txlp2_kt224{S}"))
print("--- ResNet-50 BN vs GN on mixed, per seed:", theo_seed(r"fedavg_tron_nguon_resnet50_gn.a1h_in1k_txlp1_kt160" + S, r"fedavg_tron_nguon_resnet50.a1h_in1k_txlp1_kt160" + S))
print("--- FedProx / FedAvgM")
print("fedprox mu1 effnet", m(rf"fedprox_mu1_nguon_efficientnet_b0_txlp1_gh_kt224{S}"), "fedavg", m(rf"fedavg_nguon_efficientnet_b0_txlp1_gh_kt224{S}"))
print("fedprox mu0.01 effnet", m(rf"fedprox_nguon_efficientnet_b0_txlp1_kt224{S}"), "mu0.1", m(rf"fedprox_mu0.1_nguon_efficientnet_b0_txlp1_gh_kt224{S}"))
print("convnext fedprox", m(rf"fedprox_nguon_convnext_atto_txlp1_kt224{S}"), "mu0.1", m(rf"fedprox_mu0.1_nguon_convnext_atto_txlp1_kt224{S}"),
      "fedavgm", m(rf"fedavgm_nguon_convnext_atto_txlp1_kt224{S}"))
print("--- IV-C ablation Design 1 (base convnext 3 seed)", m(rf"fedavg_nguon_convnext_atto_txlp1_kt224{S}"))
for duoi in ("khp", "cb", "tskhong", "e2", "v50"):
    v = m(rf"fedavg_nguon_convnext_atto_txlp1_{duoi}_kt224{S}")
    print(f"  {duoi}: {v}  per-seed vs base:", theo_seed(rf"fedavg_nguon_convnext_atto_txlp1_kt224{S}", rf"fedavg_nguon_convnext_atto_txlp1_{duoi}_kt224{S}"))
print("  p2:", m(rf"fedavg_nguon_convnext_atto_txlp2_kt224{S}"))
print("--- direction (cosine) logging runs")
for f in sorted(glob.glob(str(R / "lien_ket" / "*_gh_kt224_s[0-2].json"))):
    d = json.load(open(f))
    if "fedavg" not in Path(f).name or "efficientnet" not in Path(f).name or "txlp1" not in Path(f).name:
        continue
    cos = [h.get("cos") for h in d["lich_su"] if h.get("cos")]
    print(Path(f).name, "rounds", len(cos))
