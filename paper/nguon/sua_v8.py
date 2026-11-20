"""sua_v8.py — doc_v7.txt -> doc_v8.txt.
(1) Đoạn mở đầu III đi qua Hình 1 theo đường dữ liệu; mỗi dấu ①–④ nối tới một mục Results.
(2) Results: câu đầu mỗi đoạn nói kết quả; bớt số đã có trong bảng.
(3) Discussion: không lặp Results; Limitations chia ba nhóm.
(4) Thống nhất thuật ngữ: FL gap, real/mixed sites, centralized.
(5) Số đã kiểm lại: 38.5% (chia 80/20 theo ảnh), 94% theo dõi (93.8%)."""
import re
from collections import Counter
from pathlib import Path

D = Path(__file__).parent
L = [l.split("\t", 1) for l in (D / "doc_v7.txt").read_text().split("\n")]
L = [x if len(x) == 2 else [x[0], ""] for x in L]


def tim(dau, loai="P"):
    i = [k for k, (t, s) in enumerate(L) if t == loai and s.startswith(dau)]
    assert len(i) == 1, (dau, len(i))
    return i[0]


def doan(dau, moi):
    L[tim(dau)][1] = moi


def thay(dau, cu, moi, loai="P"):
    k = tim(dau, loai)
    assert L[k][1].count(cu) == 1, (dau, cu)
    L[k][1] = L[k][1].replace(cu, moi)


def xoa(dau):
    del L[tim(dau)]


def chen_sau(dau, cac_doan):
    k = tim(dau)
    L[k + 1:k + 1] = [["P", s] for s in cac_doan]


# ------------------------------------------------------------------ Abstract: định nghĩa FL gap một lần
thay("Federated learning (FL) lets hospitals", "but on real sites it is less accurate than centralized training,",
     "but on real sites it is less accurate than centralized training (the FL gap),")
thay("Federated learning (FL) lets hospitals", "recovers 35% to 49% of the accuracy gap of batch-norm (BN) models",
     "recovers 35% to 49% of the FL gap of batch-norm (BN) models")
thay("This setup comes with two hopes", "The shared model should come close to", "The shared model should come close to")

# ------------------------------------------------------------------ III: đoạn mở đầu = đi qua Hình 1
doan("Fig. 1 shows the pipeline that all experiments share.", (
    "Fig. 1 is the skeleton of the study: every experiment runs this pipeline, and experiments differ only in which "
    "variant fills each box. One round goes as follows. The Site split divides the Training images among K hospitals. "
    "Inside each hospital, the site's own images pass through Preprocessing and Augmentation into the Backbone, a stack "
    "of N blocks, each followed by a Norm layer, and then into a linear Classifier; the Local update trains this network "
    "on the site's images only. Each site then sends its model to the server, where Aggregation combines the K site "
    "models into one global model, which goes back to every site for the next round. After the last round, the Final "
    "global model is scored by the Accuracy box and attacked by the Membership attack box. Centralized training, the "
    "reference for every gap, uses the same boxes in one place that holds all images, with no server."))
chen_sau("Fig. 1 is the skeleton of the study:", [
    "A box with a general name, such as Backbone or Aggregation, has several variants; the section printed in its corner "
    "describes them, and Sections III-A to III-H follow the boxes in this order. A box with a specific label, such as the "
    "linear Classifier, is the same in every experiment. The four numbered marks are the points that the study changes "
    "or measures, and each links a part of the method to a part of the results: at ① the device-mixing control removes "
    "device differences between sites (Section IV-A); at ② the server also averages the BN statistics, the path tested "
    "in Section IV-B; at ③ a server without secure aggregation sees each site's model, and at ④ the final model is "
    "attacked (Section IV-D). Fig. 3 zooms into the DP variant of the Local update box (Section IV-E)."])
doan("Fig. 1. Pipeline shared by all experiments.", (
    "Fig. 1. Pipeline shared by all experiments; Sections III-A to III-H follow its boxes. A box with a general name has several variants, described in the section printed in its corner. Marks: "
    "① site split, real or device-mixed (Section IV-A); ② averaging at the server, including the BN statistics "
    "(Section IV-B); ③ site models seen by a server without secure aggregation, and ④ membership attacks on the final "
    "model (Section IV-D)."))

L[tim("A. Setup and metrics", "H")][1] = "A. Setup and accuracy metrics"
thay("The cost of FL is the gap to a centralized model", "The cost of FL is the gap to a centralized model with the same backbone and recipe.",
     "The FL gap is the accuracy that FedAvg on the real sites loses against a centralized model with the same backbone "
     "and recipe.")
thay("Training and validation images are split by lesion;",
     "a split by image would put a second photo of the same lesion into training for 38% of the validation images.",
     "a random 80/20 split by image would put a second photo of the same lesion into training for 38.5% of the "
     "validation images.")
thay("Each run was logged with its expected result", "if mixed minus source is above 0.03 in every seed.",
     "if FedAvg on the mixed sites beats FedAvg on the real sites by more than 0.03 in every seed.")

# ------------------------------------------------------------------ IV-A
L[tim("A. Device share of the federation gap", "H")][1] = "A. Device share of the FL gap"
doan("The real and device-mixed sites hold the same images", (
    "Removing device differences recovers 35% to 49% of the FL gap for the three BN models and none of it for the two "
    "LN models (Table III, Fig. 4). The real and device-mixed sites (mark ① in Fig. 1) hold the same images with the "
    "same class counts, so the change comes from devices alone. For BN models, FedAvg loses 0.10 to 0.14 against "
    "centralized training, and mixing wins back about 0.05. For LN models the gap is smaller (0.076 and 0.036), and "
    "mixing does not help: ConvNeXt-atto changes by -0.004, and DeiT-tiny drops by 0.044 and stays below the real "
    "sites in all three mixing draws."))
thay("TABLE III.", "recovers part of the federation gap for BN models only", "recovers part of the FL gap for BN models only")
thay("TABLE III.", "Gap = central minus FedAvg source; device share = (mixed − source) / gap;",
     "Gap = centralized minus FedAvg real; device share = (FedAvg mixed − FedAvg real) / gap;")
k = tim("TABLE III.")
for j in range(k + 1, k + 8):
    L[j][1] = {"Central": "Centralized", "FedAvg source": "FedAvg real"}.get(L[j][1], L[j][1])
doan("The smaller gap also gives a better federated model.", (
    "The LN model with the smaller gap is also the best federated model. With five seeds, FedAvg reaches 0.741 +/- "
    "0.013 with ConvNeXt-atto and 0.661 +/- 0.010 with EfficientNet-B0 (mean +/- standard deviation), and the seed "
    "ranges do not overlap (0.719 to 0.752 against 0.646 to 0.672); on the test set the two score 0.662 and 0.597. "
    "Centralized, ConvNeXt-atto is ahead by only 0.035 (Table III), so most of the difference appears only in FL."))

# ------------------------------------------------------------------ IV-B
doan("The rest of the gap, about 60% for BN models", (
    "No design we tried reduced the rest of the gap, about 60% for BN models and nearly all of it for LN models, which "
    "remains when only disease mix and site size differ. FedProx with mu = 1 cut the drift of the largest site by a "
    "factor of three without improving accuracy (+0.009), and neither FedProx nor FedAvgM helped ConvNeXt-atto (FedAvgM "
    "fell to 0.597). Per site, only rosendahl has a gap whose bootstrap interval excludes zero ([0.04, 0.21]); the point "
    "gap of vienna_dias (0.31) rests on 74 validation images and is not significant."))

# ------------------------------------------------------------------ IV-C
doan("Each final design is tested by changing one part at a time", (
    "Table IV keeps every part of both final designs that passes the rule of Section III-G. Design 1 is ConvNeXt-atto "
    "with FedAvg, 30 rounds of one local epoch, global class weights and P1. Removing ImageNet pretraining, switching to "
    "EfficientNet-B0 or using local class weights lowers accuracy in every seed, and FedAvgM is much worse than FedAvg; "
    "the other changes are within noise. Global class weights therefore stay, at the cost that each hospital reports "
    "its number of cases per class."))
doan("Design 2 is a frozen EfficientNet-B0 with DP-SGD on the classifier (L0), 30 rounds and P1;", (
    "Design 2 is a frozen EfficientNet-B0 with DP-SGD on the classifier (L0), 30 rounds and P1, with the learning rate "
    "chosen on validation for each epsilon. At epsilon = 3, 10 rounds and a ConvNeXt-atto backbone are worse in every "
    "seed. The proposed design is Design 2 with BCN weights in place of ImageNet weights (Section IV-E)."))

# ------------------------------------------------------------------ IV-D
doan("A low leakage figure can mean a private model or a weak attack,", (
    "Federated models leak less than centralized ones, also under a strong attack (Table V, mark ④ in Fig. 1). RMIA "
    "finds four to twelve times more members than the loss attack at a 1% false-positive rate, so the low figures are "
    "not due to a weak attack. Under RMIA the true-positive rate is 0.043 for federated against 0.096 for centralized "
    "EfficientNet-B0, and 0.054 against 0.071 for ConvNeXt-atto; the two federated models leak about equally."))
doan("A curious server that sees each site's model before averaging", (
    "A curious server that sees each site's model before averaging learns more than one that sees only the average "
    "(mark ③ in Fig. 1). For EfficientNet-B0, the rosendahl model gives an RMIA AUC of 0.672 and a true-positive rate "
    "of 0.118, against 0.577 and 0.044 for the averaged model; for vidir_modern the AUC rises from 0.622 to 0.676. With "
    "the smaller learning rate of the best round the direction is the same and the differences are smaller (rosendahl "
    "0.585 to 0.658). For ConvNeXt-atto the difference is small or absent. For EfficientNet-B0 the averaged model "
    "leaks about as much as the global model before the round."))

# ------------------------------------------------------------------ IV-E
doan("Clipping alone costs little; most of the cost of DP comes from the noise.", (
    "DP works only on the classifier (L0). Without DP, training deeper layers helps (the last two blocks of "
    "ConvNeXt-atto reach 0.702), but with DP at L1 and L2 accuracy falls to between 0.15 and 0.24, near chance (0.14), "
    "at every epsilon, also with a batch of 1024 and a learning rate of 3e-4. At L0 most of the cost comes from the "
    "noise: with ImageNet features, clipping alone lowers accuracy from 0.617 to 0.602, and the noise at epsilon = 3 "
    "lowers it to 0.430 (Table VI). EfficientNet-B0 is the best of eight frozen ImageNet backbones at epsilon = 3, and "
    "ConvNeXt-atto stays below it at every epsilon (0.409, 0.353 and 0.276 at epsilon = 8, 3 and 1), so hypothesis A "
    "is rejected."))
doan("With backbone weights pretrained on public BCN images instead of ImageNet,", (
    "Public dermoscopy features raise DP accuracy by 0.12 to 0.13 at every epsilon (Table VI, Fig. 5). Without DP, a "
    "classifier on BCN features (0.710) already beats full FedAvg training of EfficientNet-B0 (0.661). At epsilon = 3 "
    "the proposed design reaches 0.554. This passes the threshold of 0.55 set in advance, but by less than the spread "
    "between seeds, so we count it as a clear gain over ImageNet features rather than a sure pass of the threshold. It "
    "is still 0.19 below Design 1. The gain does not come from shared images: the perceptual hash flagged 7,463 "
    "candidate pairs between BCN and HAM10000, and none was a confirmed duplicate."))

# ------------------------------------------------------------------ IV-F
doan("With the long budget and a matched 60-epoch centralized run,", (
    "The device effect repeats on Fed-ISIC2019 at a similar size with the long budget, 27 rounds of 100 steps against "
    "60 centralized epochs. For EfficientNet-B0 the gap is 0.137 (0.777 to 0.640), and mixing raises FedAvg by 0.057, "
    "consistent across seeds, a device share of 42% against 40% on HAM10000. For ConvNeXt-atto the gap is 0.053 (0.736 "
    "to 0.683), and mixing adds only 0.008. On all test images together the two gains are 0.048 and 0.030, so the LN "
    "model is less affected but not untouched: per centre, mixing helps it at MSK and rosendahl and hurts it at "
    "vidir_modern and vienna_dias. Centralized, EfficientNet-B0 is the stronger model (0.777 against 0.736), mostly "
    "because of one centre, vidir_modern (0.858 against 0.681)."))

# ------------------------------------------------------------------ V Discussion
doan("On real hospitals, about 40% of the FL gap for BN models", (
    "For hospitals that build a federated skin-lesion model, the first choice is the normalization of the backbone. "
    "With BN, part of the accuracy lost to FL comes from device differences between hospitals, and an LN backbone "
    "avoids that part without needing the source of each image, which FedBN needs. Colour constancy and FedBN reduce "
    "the same loss for BN models, but less (Section IV-B). The rest of the gap, which comes with differences in disease "
    "mix and site size, was not reduced by any design we tested and is the main open problem."))
doan("The Fed-ISIC2019 results also hold a lesson on method.", (
    "Benchmarks for FL should state their training budget and their metric. On Fed-ISIC2019 the benchmark budget hid "
    "the LN advantage (Section IV-F); had we stopped there, we would have concluded that it was specific to HAM10000. "
    "The lead also holds only for the per-centre mean, so comparisons on this benchmark should report both metrics."))
doan("On privacy, federated training lowers membership leakage", (
    "Keeping images on site is not enough to protect patients. Federated models leak less than centralized ones but "
    "still leak, and a server that sees each site's model learns more than one that sees only their average, so secure "
    "aggregation is worth its cost, at least for BN models. Formal protection with DP is practical at this scale only on "
    "the last layer, and then it depends on the frozen features: public images from another hospital give better "
    "features than ImageNet. A hospital must still choose between Design 1, more accurate but with only measured "
    "protection, and the DP design, 0.19 less accurate but with a formal guarantee."))
doan("These conclusions have limits.", "These conclusions have three kinds of limits.")
chen_sau("These conclusions have three kinds of limits.", [
    "RUNIN:Scale.|The federation is small, four sites and 8,583 training images, and the smallest site has 74 "
    "validation images, so most per-site differences are not significant. The 4 GB GPU limited the ResNet-50 pair to "
    "160 pixels and ruled out larger backbones, and the public backbone was pretrained with one seed.",
    "RUNIN:Mechanism.|The four comparisons point to the BN statistics but do not prove that they are the only path. "
    "Beyond the ResNet-50 pair, LN and BN models also differ in architecture, so part of the LN advantage may come from "
    "the architecture itself. The device share depends on how the mixed sites are built; the LN result holds on "
    "average, not at every site, and the drop of DeiT-tiny under mixing is unexplained. The rest of the gap mixes "
    "differences in disease mix and size with the cost of FL itself, and a control with identically distributed sites "
    "would be needed to separate them. We did not study why EfficientNet-B0 is stronger centrally on Fed-ISIC2019.",
    "RUNIN:Privacy.|RMIA uses reference models trained centrally while the targets are trained with FL, so the measured "
    "leakage is a lower bound. DP is counted per image, with each site spending its own budget; DP at the level of whole "
    "sites is not practical with four sites. BCN belongs to the same ISIC collection as HAM10000, and choosing the "
    "learning rate on validation makes validation scores slightly optimistic, so test scores are reported beside them. "
    "Source inference, gradient inversion and malicious sites are out of scope."])

thay("We measured what federated learning costs", "imaging devices cause about 40% of the accuracy gap", "imaging devices cause about 40% of the FL gap")

xoa("Section II reviews related work, Section III describes the methods")

thay("FedAvg [2] averages the weights", " FedProx [4] and server momentum (FedAvgM) [3] change the optimization, and we test both.", "")
thay("For hospitals that build a federated", " Colour constancy and FedBN reduce the same loss for BN models, but less (Section IV-B).", "")
doan("Future work should target", "Future work should target the part of the gap caused by disease mix and site size, and test longer budgets and larger backbones on more federations.")
thay("FOOTNOTE:All experiments", " Runs resume from checkpoints and log their settings, code hash, data-split hash and learning curves.", "")

thay("Each image has one of C = 7 labels.", "of the n images. In each round", "of the n images (on Fed-ISIC2019, C = 8 and K = 6). In each round")

# ------------------------------------------------------------------ kiểm thuật ngữ và số
toan = " ".join(s for t, s in L)
for cu in ("federation gap", "accuracy gap", "FedAvg source", "mixed minus source", "central minus"):
    if cu in toan: print("CÒN:", cu, [s[:90] for t, s in L if cu in s])
so = lambda txt: Counter(re.findall(r"\d+(?:[.,]\d+)*%?", txt))
cuoi = lambda X: " ".join(s for t, s in X[:[k for k, (t, s) in enumerate(X) if s == "References"][0]])
L7 = [l.split("\t", 1) for l in (D / "doc_v7.txt").read_text().split("\n")]
L7 = [x if len(x) == 2 else [x[0], ""] for x in L7]
print("số bỏ đi so với v7:", dict(so(cuoi(L7)) - so(cuoi(L))))
print("số thêm so với v7:", dict(so(cuoi(L)) - so(cuoi(L7))))
(D / "doc_v8.txt").write_text("\n".join(f"{t}\t{s.strip()}" for t, s in L))
print("ok")
