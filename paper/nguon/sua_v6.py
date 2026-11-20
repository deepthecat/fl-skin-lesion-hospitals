"""sua_v6.py — doc.txt (rev 96) -> doc_v6.txt: sửa số sai, viết lại Abstract, câu đầu IV-B/IV-F, bảng ablation mới,
đánh số lại bảng, chú thích hình mới. Mỗi thay đổi tìm theo đầu đoạn và assert đúng một lần."""
import re
from pathlib import Path

D = Path(__file__).parent
L = [l.split("\t", 1) for l in (D / "doc.txt").read_text().split("\n")]
L = [x if len(x) == 2 else [x[0], ""] for x in L]


def tim(dau, loai="P"):
    i = [k for k, (t, s) in enumerate(L) if t == loai and s.startswith(dau)]
    assert len(i) == 1, (dau, len(i))
    return i[0]


def thay_doan(dau, moi):
    L[tim(dau)][1] = moi


def thay_trong(dau, cu, moi):
    k = tim(dau)
    assert L[k][1].count(cu) == 1, (dau, cu)
    L[k][1] = L[k][1].replace(cu, moi)


def xoa_doan(dau):
    del L[tim(dau)]


# ------------------------------------------------------------------ Abstract
thay_doan("Federated learning (FL) lets hospitals", (
    "Federated learning (FL) lets hospitals train a shared skin-lesion classifier without exchanging images, but on real "
    "sites it loses accuracy against centralized training, and the trained model can still reveal which images it was "
    "trained on. Studies usually report the accuracy loss as one average over sites and do not measure leakage. We measure "
    "both on HAM10000, whose four acquisition sources act as hospitals, and repeat the key experiments on the six centres "
    "of Fed-ISIC2019. Removing device differences between sites while keeping their images and disease mix recovers 35% "
    "to 49% of the accuracy gap of batch-norm (BN) models on HAM10000 and 28% to 42% on Fed-ISIC2019, and four controlled "
    "comparisons point to the averaged BN statistics as the path. Mixing does not help layer-norm models, and ConvNeXt-atto "
    "gives the best federated model (0.741 balanced accuracy, against 0.661 for EfficientNet-B0), although on Fed-ISIC2019 "
    "its lead needs three times the benchmark's training steps and holds only in the per-centre metric. Under a strong "
    "membership attack (RMIA), federated models leak less than centralized ones, but a server that sees each site's BN "
    "model learns more than one that sees only their average. Differential privacy is usable only on the last layer; "
    "pretraining the frozen backbone on public dermoscopy images from another hospital raises its accuracy at epsilon = 3 "
    "from 0.43 to 0.55, still 0.19 below the best non-private model."))

# ------------------------------------------------------------------ Introduction: gộp đoạn 5 + 6, câu thông điệp chung
thay_doan("We study both questions on HAM10000 [11]", (
    "We study both questions on HAM10000 [11], whose four acquisition sources act as hospitals, and repeat the key "
    "experiments on the six centres of Fed-ISIC2019. The main claims rest on comparisons in which only one factor changes: "
    "a device-mixing control keeps each hospital's size and disease mix but removes device differences; a pair of ResNet-50 "
    "models differs only in the normalization layer; and on Fed-ISIC2019, architectures are compared at the benchmark budget "
    "and at three times as many training steps. Leakage is measured with RMIA [12], a strong membership-inference attack, "
    "also for a server that sees each hospital's model before averaging, and DP is applied at three depths of the network. "
    "In short, we measure what FL costs on real hospitals in accuracy and in privacy, trace part of the accuracy cost to "
    "device differences that enter through batch-normalization statistics, and reduce part of the privacy cost with public "
    "features under DP."))
xoa_doan("Both losses can be partly traced and reduced.")

# ------------------------------------------------------------------ III: đoạn mở đầu + chú thích Hình 1
thay_doan("Fig. 1 shows the workflow.", (
    "Fig. 1 shows one federated run from the input images to the evaluation; the numbered marks show where the study "
    "intervenes or measures. The training images are split into four sites, either by source or by the device-mixing "
    "control (①). Each site trains the same network on its own images, and the server averages all weights, including the "
    "BN statistics (②). Without secure aggregation the server sees each site's model before averaging (③), and the final "
    "global model is scored for accuracy and attacked for membership (④). The DP designs keep the backbone frozen and train "
    "only the classifier with DP-SGD (Fig. 3)."))
thay_doan("Fig. 1. Workflow of the study.", (
    "Fig. 1. Pipeline of one federated run on HAM10000. The four hospitals train the same network on their own images, "
    "and the server averages all weights, including the BN statistics. Marks: ① real or device-mixed site split "
    "(Section III-D); ② averaged BN statistics, the path tested in Section IV-B; ③ site models visible to a server "
    "without secure aggregation (Section IV-D); ④ membership attacks on the final model (Section III-E)."))
thay_doan("Fig. 3. Proposed design.", (
    "Fig. 3. Proposed design. EfficientNet-B0 is pretrained on public BCN images and frozen; each hospital computes its "
    "features once and trains only the linear classifier with DP-SGD, and the server averages the classifier weights. "
    "The design differs from Design 2 only in the frozen weights: BCN here, ImageNet there."))

# ------------------------------------------------------------------ III-G: đoạn GPU thành chú thích cuối trang
k = tim("All experiments run on one laptop GPU")
L[k][1] = "FOOTNOTE:" + L[k][1]

# ------------------------------------------------------------------ Bảng III
cu = [i for i, (t, s) in enumerate(L) if t == "C"]
def o(gia_tri_cu, moi, sau=None):
    ung = [i for i in cu if L[i][1] == gia_tri_cu and (sau is None or L[i - 5][1] == sau or L[i - 6][1] == sau)]
    assert len(ung) == 1, (gia_tri_cu, len(ung))
    L[ung[0]][1] = moi
o("0.119", "0.118")
o("~0", "−5%")
o("negative", "n/a")
thay_trong("TABLE III.", "Mixed values use one draw",
           "n/a: mixing lowered accuracy by more than the gap. Mixed values use one draw")

# ------------------------------------------------------------------ IV-A: ± = độ lệch chuẩn
thay_doan("The smaller gap also gives a better federated model.", (
    "The smaller gap also gives a better federated model. With five seeds, FedAvg reaches 0.741 +/- 0.013 with "
    "ConvNeXt-atto and 0.661 +/- 0.010 with EfficientNet-B0 (mean +/- standard deviation), and the seed ranges do not "
    "overlap (0.719 to 0.752 against 0.646 to 0.672); on the test set the two score 0.662 and 0.597. Centralized, "
    "ConvNeXt-atto is ahead by only 0.035 (Table III), so most of the difference appears only under federation."))

# ------------------------------------------------------------------ IV-B: câu đầu nói kết quả, 0.048 -> 0.047
thay_doan("Two mechanisms were stated in advance:", (
    "Four results point to the BN statistics that FedAvg averages (M2, mark ② in Fig. 1), and none to the direction of "
    "local updates (M1). Mixing raised FedAvg accuracy by 0.047 for EfficientNet-B0 while the cosine similarity between "
    "site updates changed by at most 0.05, so M1 is rejected. For M2, first, mixing helps all three BN models and neither "
    "LN model. Second, in the ResNet-50 pair, mixing gains 0.051 with BN, consistent across seeds, and 0.014 with GN, "
    "within noise. Third, FedBN, which keeps BN statistics local, gains 0.080 over FedAvg for EfficientNet-B0 and 0.040 "
    "for ResNet-18 on the real sites, but only 0.013 and 0.006 on the mixed sites. Fourth, colour constancy (P2) leaves "
    "centralized accuracy unchanged and raises FedAvg by 0.024 for EfficientNet-B0 but only 0.006 for ConvNeXt-atto."))

# ------------------------------------------------------------------ IV-C: bảng ablation mới (Table IV)
k = tim("Each final design is tested by removing one component")
L[k][1] = ("Each final design is tested by changing one component at a time, with the rule of Section III-F (Table IV). "
           "Design 1 is ConvNeXt-atto with FedAvg, 30 rounds of one local epoch, global class weights and P1. ImageNet "
           "pretraining, the ConvNeXt-atto backbone and global class weights pass the rule, and server momentum (FedAvgM) "
           "is clearly worse than FedAvg; the other changes are within noise. Global class weights therefore stay, at the "
           "cost that each hospital reports its case counts per class.")
k2 = tim("Design 2 is a frozen EfficientNet-B0 with DP-SGD")
L[k2][1] = ("Design 2 is a frozen EfficientNet-B0 with DP-SGD on the classifier (L0), 30 rounds and P1; its learning rate is "
            "chosen on validation for each epsilon. At epsilon = 3, 30 rounds and the EfficientNet-B0 backbone pass the "
            "rule against 10 rounds and ConvNeXt-atto, while 60 rounds, P2 and a ViT-S backbone are within noise. The "
            "proposed design (Section IV-E) is Design 2 with the ImageNet weights of the frozen backbone replaced by "
            "weights pretrained on public BCN images.")
BANG = [("Change", "Val.", "Diff.", "Rule met"),
        ("Design 1: ConvNeXt-atto, FedAvg", "0.737", "", ""),
        ("no ImageNet pretraining", "0.377", "−0.360", "yes"),
        ("EfficientNet-B0 backbone", "0.659", "−0.078", "yes"),
        ("local class weights", "0.706", "−0.031", "yes"),
        ("no class weights", "0.716", "−0.021", "no"),
        ("FedAvgM (server momentum)", "0.597", "−0.141", "yes"),
        ("FedProx (mu = 0.01)", "0.741", "+0.004", "no"),
        ("two local epochs", "0.742", "+0.005", "no"),
        ("P2 colour constancy", "0.743", "+0.006", "no"),
        ("50 rounds", "0.747", "+0.010", "no"),
        ("Design 2: DP at L0, epsilon = 3", "0.430", "", ""),
        ("10 rounds", "0.345", "−0.085", "yes"),
        ("ConvNeXt-atto backbone", "0.353", "−0.077", "yes"),
        ("ViT-S backbone", "0.418", "−0.012", "no"),
        ("60 rounds", "0.424", "−0.006", "no"),
        ("P2 colour constancy", "0.432", "+0.003", "no")]
moi = [["P", "TABLE IV. Ablation of the two final designs. Each row changes one component and keeps the rest of the "
             "design. Validation balanced accuracy, mean of three seeds; Diff. = change from the design; rule met = lower "
             "by more than 0.02 in every seed (Section III-F)."]]
moi += [["C", c] for r in BANG for c in r]
L[k2 + 1:k2 + 1] = moi

# ------------------------------------------------------------------ IV-E: cắt gradient không phải "miễn phí"
thay_trong("Clipping alone costs nothing;",
           "Clipping alone costs nothing; the cost of DP comes from the noise. Training deeper layers",
           "Clipping alone costs little; most of the cost of DP comes from the noise. With ImageNet features at L0, "
           "clipping lowers accuracy from 0.617 to 0.602, and the noise at epsilon = 3 lowers it further to 0.430 (Table V). "
           "Training deeper layers")
thay_trong("Clipping alone costs little;", "accuracy falls to chance at every epsilon,",
           "accuracy falls to between 0.15 and 0.24, near chance (0.14), at every epsilon,")
thay_trong("Clipping alone costs little;", "It still pays heavily, falling from 0.617 without DP to 0.430 at epsilon = 3 (Table V).",
           "")

# ------------------------------------------------------------------ IV-F
thay_doan("Under the FLamby protocol on Fed-ISIC2019", (
    "Our reproduction of the FLamby protocol on Fed-ISIC2019 agrees with the original results within their spread. "
    "We obtain 0.731 centralized, 0.624 with FedAvg and 0.652 with FedProx, against 0.708 +/- 0.016, 0.539 +/- 0.097 and "
    "0.544 +/- 0.074 (mean +/- standard deviation over five seeds) in the results released with FLamby [9]. The per-centre "
    "pattern also repeats: rosendahl is the weakest centre even when centralized (0.630; FLamby 0.623), and MSK drops most "
    "under FedAvg (0.415; FLamby 0.387). Under this budget, device mixing raises FedAvg by 0.030 (device share 28%), "
    "exactly at the replication threshold."))
thay_trong("The ranking of the two architectures depends on the budget",
           "where EfficientNet-B0 stops improving within ten rounds",
           "where the mean curve of EfficientNet-B0 levels off within ten rounds")
thay_trong("Fig. 6. EfficientNet-B0 stops improving", "EfficientNet-B0 stops improving within the first ten rounds while ConvNeXt-atto keeps learning.",
           "EfficientNet-B0 levels off within ten rounds while ConvNeXt-atto keeps improving.")

# ------------------------------------------------------------------ VI Conclusion: lặp câu thông điệp, nêu số hai thiết kế
thay_doan("We measured accuracy and privacy together", (
    "We measured what federated learning costs on real hospitals in accuracy and in privacy, on the four sources of "
    "HAM10000 and the six centres of Fed-ISIC2019. For BN models, acquisition devices cause about 40% of the accuracy gap, "
    "and four comparisons point to the averaged BN statistics as the path; an LN backbone avoids this part, while its "
    "advantage on Fed-ISIC2019 depends on the training budget and the metric. Federated training leaks less than "
    "centralized training, individual site models leak more than their average, and DP is usable only at the last layer, "
    "where public dermoscopy features recover part of its cost. At this scale no single design wins on both axes: Design 1 "
    "reaches 0.741 without formal guarantees, and the proposed DP design reaches 0.554 at epsilon = 3."))


# ------------------------------------------------------------------ III-E / IV-D: số liệu server tò mò và theo lớp (đối chiếu sim_may_chu_v2.csv, tan_cong_theo_site.csv)
thay_trong("Without secure aggregation [27] the server sees", "We simulate one round at the best round of the trained FedAvg runs and attack",
           "We simulate one extra round starting from the best global model of each FedAvg run. The cosine schedule makes "
           "the learning rate of that late round very small, so local training uses the learning rate of round 10, when "
           "updates have their typical size. We attack")
thay_doan("Leakage differs more between classes than between sites.", (
    "Leakage differs more between classes than between sites. For the centralized EfficientNet-B0 under the loss attack, "
    "AKIEC reaches an AUC of 0.872 and nevi 0.536 (AKIEC is above 0.84 and nevi below 0.55 in every seed), while across "
    "sites the AUC ranges only from 0.525 at vidir_molemax to 0.700 at rosendahl, and the smallest site, vienna_dias "
    "(0.623), leaks less than rosendahl and vidir_modern. H2 therefore holds for classes and is rejected for sites. Across "
    "the seven classes, leakage follows the train-validation loss gap (Spearman 0.75) more than class size (-0.46): it "
    "tracks memorization rather than rarity, and VASC is rare but leaks little."))
thay_doan("A curious server that sees each site's model before averaging", (
    "A curious server that sees each site's model before averaging learns more than one that sees only the aggregate "
    "(mark ③ in Fig. 1). For EfficientNet-B0, the rosendahl model gives an RMIA AUC of 0.672 and a true-positive rate of "
    "0.118, against 0.577 and 0.044 for the aggregated model; for vidir_modern the AUC rises from 0.622 to 0.676. With the "
    "smaller learning rate of the best round the direction is the same and the differences are smaller (rosendahl 0.585 "
    "to 0.658). For ConvNeXt-atto the difference is small or absent. For EfficientNet-B0 the aggregated model leaks about "
    "as much as the global model before the round, so with secure aggregation the server learns little beyond what any "
    "holder of the final model learns."))
thay_doan("TABLE II. HAM10000 sources used as federated sites", "TABLE II. HAM10000 sources used as federated sites (lesion-level split; share of NV among the training images).")

thay_trong("The Fed-ISIC2019 results also carry a lesson on method.", "because EfficientNet-B0 stops improving early", "because EfficientNet-B0 levels off early")

# ------------------------------------------------------------------ đánh số lại bảng IV->V, V->VI, VI->VII (bảng mới là IV)
for i, (t, s) in enumerate(L):
    if s.startswith("TABLE IV. Ablation"):
        continue
    s = re.sub(r"\bTable VI\b", "Table @7", s); s = re.sub(r"\bTable V\b", "Table @6", s)
    s = re.sub(r"\bTable IV\b", "Table @5", s) if not s.startswith("Each final design is tested") else s
    s = re.sub(r"^TABLE VI\.", "TABLE @7.", s); s = re.sub(r"^TABLE V\.", "TABLE @6.", s); s = re.sub(r"^TABLE IV\.", "TABLE @5.", s)
    L[i][1] = s.replace("@7", "VII").replace("@6", "VI").replace("@5", "V")

(D / "doc_v6.txt").write_text("\n".join(f"{t}\t{s.strip()}" for t, s in L))
print("ok", sum(1 for t, s in L if s.startswith("TABLE")), "tables")
for t, s in L:
    if s.startswith("TABLE"):
        print("  ", s[:60])
