"""sua_v9.py — viết lại toàn bộ paper (doc_v9.txt) theo mạch đã chốt 10.11.2026:
mạch chính = hiệu năng (thiết bị -> BN -> LN -> thiết kế tốt nhất so với baseline FLamby, lặp lại Fed-ISIC2019);
mạch phụ duy nhất = riêng tư (vẫn rò -> DP chỉ ở lớp cuối -> đặc trưng công khai -> đánh đổi).
Mỗi thành phần của con đường chính có lý do ngay tại chỗ. Trích dẫn viết [@khoá], đánh số theo thứ tự xuất hiện.
Mọi con số lấy từ doc_v8 (đã kiểm với JSON/CSV) hoặc recall_theo_lop.csv (chạy 10.11.2026)."""
import json
import re
from pathlib import Path

D = Path(__file__).parent
CU = [l.split("\t", 1) for l in (D / "doc_v8.txt").read_text().split("\n")]
REF_CU = {}
for t, s in CU:
    m = re.match(r"^\[(\d+)\]\s*(.*)$", s)
    if m:
        REF_CU[int(m.group(1))] = m.group(2)
KHOA = dict(rieke=1, mcmahan=2, hsu=3, fedprox=4, shokri=5, yeom=6, abadi=8, flamby=9, chen=10, ham=11, rmia=12,
            fedbn=13, hsieh=14, gn=15, qu=16, siomos=17, codella=18, cassidy=19, bissoto=20, fogelberg=21, sog=22,
            carlini=25, bonawitz=27, tramer=28, ziller=29, tan=31, he=32, liu=33, touvron=34, loshchilov=35)

M = []
H = lambda s: M.append(["H", s])
P = lambda s: M.append(["P", s])
HINH = lambda s: M.extend([["P", ""], ["P", s]])


def BANG(tieu_de, hang):
    M.append(["P", tieu_de])
    for h in hang:
        for c in h:
            M.append(["C", c])


# ================================================================== Abstract
H("Abstract")
P("Federated learning (FL) lets hospitals train a shared skin-lesion classifier without sharing images, but on real "
  "sites it is less accurate than centralized training, and the trained model can still reveal which images it was "
  "trained on. We ask where the accuracy goes and what formal privacy costs, using the four image sources of HAM10000 "
  "as hospitals and the six centres of Fed-ISIC2019 as a second test. A control that removes device differences "
  "between sites, while keeping each site's images and disease mix, recovers 35% to 49% of the FL gap of batch-norm "
  "(BN) models, and four controlled comparisons point to the BN statistics that the server averages as the reason. A "
  "layer-norm backbone avoids this loss: ConvNeXt-atto reaches 0.741 balanced accuracy against 0.661 for the "
  "EfficientNet-B0 baseline of the FLamby benchmark and keeps most of its melanoma recall, and the device effect "
  "repeats on Fed-ISIC2019 (42%). For privacy, federated models still leak under a strong membership attack, and "
  "differential privacy works only on the last layer; pretraining the frozen backbone on public dermoscopy images from "
  "another hospital raises its accuracy at epsilon = 3 from 0.43 to 0.55, still 0.19 below the best non-private model.")
P("Index Terms: federated learning, batch normalization, differential privacy, membership inference, skin lesion "
  "classification, data heterogeneity.")

# ================================================================== I
H("I. Introduction")
P("A skin-cancer classifier needs many images of every lesion type, taken with many devices. No single clinic has "
  "that much data, and patient images usually cannot leave the hospital [@rieke]. Federated learning (FL) [@mcmahan] "
  "lets hospitals train one model together: each site trains on its own images, and only model weights go to a "
  "server, which averages them.")
P("On real hospitals this costs accuracy, and keeping images on site is not a formal privacy guarantee. Sites differ "
  "in patients, disease mix and imaging devices, and the averaged model is less accurate than one trained on all "
  "images in one place [@hsu], [@fedprox]. A trained model can also reveal whether an image was in its training set "
  "[@shokri]; differential privacy (DP) [@abadi] puts a formal limit on this, at a cost in accuracy.")
P("Skin-lesion FL studies, including the Fed-ISIC2019 benchmark of FLamby [@flamby] and the study of Chen et al. "
  "[@chen], report the accuracy loss as one number averaged over sites. One number does not say which difference "
  "between hospitals causes the loss, and so which fix to try. Privacy is argued from the design rather than measured "
  "with an attack.")
P("We trace the accuracy loss on HAM10000 [@ham], whose four image sources act as hospitals, and repeat the key test "
  "on Fed-ISIC2019. A device-mixing control builds a second set of sites from the same images, with the same class "
  "counts per site but with devices mixed across sites, so any change in accuracy comes from devices alone. Comparing "
  "backbones with batch normalization (BN) and layer normalization (LN), and a ResNet-50 pair that differs only in "
  "this layer, shows where the device effect enters: through the BN statistics that the server averages. An LN "
  "backbone avoids it and gives the best federated model. We then add privacy to this design: we measure leakage with "
  "a strong attack, RMIA [@rmia], and test DP-SGD at three depths of the network, with ImageNet features and with "
  "features learned from public dermoscopy images.")
P("This work makes three contributions.")

# ================================================================== II
H("II. Related Work")
P("FedAvg [@mcmahan] loses accuracy when sites hold different data [@hsu]. FedProx [@fedprox] changes local "
  "training, and FedBN [@fedbn] keeps the batch-normalization layers at each site. Hsieh et al. [@hsieh] showed that BN "
  "suffers when labels are skewed across sites and that group normalization (GN) [@gn] helps; Transformers with layer "
  "normalization coped better with different clients [@qu], and normalization-free ResNets beat BN ResNets in FL, "
  "including on Fed-ISIC2019 [@siomos]. In skin images, models learn signs of the imaging device [@bissoto] and lose "
  "accuracy on new centres [@fogelberg], but these results come from centralized training. Table I compares the "
  "closest studies with this work.")
P("FLamby [@flamby] released Fed-ISIC2019, which splits ISIC 2019 into six centres by imaging system and trains "
  "EfficientNet-B0 with FedAvg for a fixed, short budget. We use this model and protocol as our baseline: we "
  "reproduce it, and every gain we report is measured against it.")
P("Membership inference asks whether a record was in a model's training data [@shokri]. Carlini et al. [@carlini] "
  "argued for reporting the true-positive rate at a low false-positive rate, and RMIA [@rmia] gives a strong attack "
  "with a few reference models. DP-SGD [@abadi] clips the gradient of each example and adds noise; Tramèr and Boneh "
  "[@tramer] showed that under DP a linear classifier on good fixed features is a strong baseline. Ziller et al. "
  "[@ziller] measured the cost of DP on medical images, including HAM10000, in centralized training.")
BANG("TABLE I. Closest studies and this work. Real sites: clients are real hospitals or imaging sources. Gap split: "
     "the FL accuracy gap is divided by cause; partly: only some experiments use real sites. Each row describes what "
     "the study reports.",
     [["Study", "Data", "Real sites", "Gap split", "Normalization", "Attack", "DP"],
      ["FLamby [@flamby]", "Fed-ISIC2019 and others", "yes", "no", "no", "no", "no"],
      ["Chen et al. [@chen]", "dermoscopy sources", "yes", "no", "no", "no", "no"],
      ["FedBN [@fedbn]", "digits, objects, ABIDE", "partly", "no", "BN kept local", "no", "no"],
      ["Hsieh et al. [@hsieh]", "label-skew splits", "partly", "no", "BN, GN", "no", "no"],
      ["Siomos et al. [@siomos]", "Fed-ISIC2019 and others", "yes", "no", "BN, GN, none", "no", "yes"],
      ["Ziller et al. [@ziller]", "HAM10000 and others", "no", "no", "no", "reconstruction", "yes"],
      ["This work", "HAM10000, Fed-ISIC2019", "yes", "device vs rest", "BN, GN, LN", "membership", "yes"]])

# ================================================================== III
H("III. Methodology")
P("Fig. 1 is the skeleton of the study: every experiment runs this pipeline, and experiments differ only in which "
  "variant fills each box. One round goes as follows. The Site split divides the Training images among K hospitals. "
  "Inside each hospital, the site's own images pass through Preprocessing and Augmentation into the Backbone, a stack "
  "of N blocks, each followed by a Norm layer, and then into a linear Classifier; the Local update trains this network "
  "on the site's images only. Each site then sends its model to the server, where Aggregation combines the K site "
  "models into one global model, which goes back to every site for the next round. After the last round, the Final "
  "global model is scored by the Accuracy box and attacked by the Membership attack box. Centralized training, the "
  "reference for every gap, uses the same boxes in one place that holds all images, with no server.")
P("A box with a general name, such as Backbone or Aggregation, has several variants; the section printed in its corner "
  "describes them, and Sections III-A to III-G follow the boxes in this order. A box with a specific label, such as the "
  "linear Classifier, is the same in every experiment. The four numbered marks are the points that the study changes "
  "or measures: at ① the device-mixing control removes device differences between sites (Section IV-A); at ② the "
  "server also averages the BN statistics, the path tested in Section IV-B; at ③ a server without secure aggregation "
  "sees each site's model, and at ④ the final model is attacked (Section IV-E). Fig. 3 zooms into the DP variant of "
  "the Local update box.")
HINH("Fig. 1. Pipeline shared by all experiments; Sections III-A to III-G follow its boxes. A box with a general name "
     "has several variants, described in the section printed in its corner. Marks: ① site split, real or "
     "device-mixed (Section IV-A); ② averaging at the server, including the BN statistics (Section IV-B); ③ site "
     "models seen by a server without secure aggregation, and ④ membership attacks on the final model (Section IV-E).")

H("A. Setup and accuracy metrics")
P("Each image has one of C = 7 labels. The training data are held by K = 4 sites, and site k holds n_k of the n images "
  "(on Fed-ISIC2019, C = 8 and K = 6). In each round the server sends the global weights to every site, each site "
  "trains on its own images, and the server averages the results in proportion to site size:")
P("Always predicting nevus gives about 67% accuracy, so the Accuracy box uses balanced accuracy, the mean recall over "
  "classes, which is also the ISIC 2018 metric [@codella]:")
P("The FL gap is the accuracy that FedAvg on the real sites loses against a centralized model with the same backbone "
  "and recipe. The device share is the accuracy that FedAvg gains when the real sites are replaced by device-mixed "
  "sites (Section III-B), divided by the gap.")
P("Training and validation images are split by lesion, because the same lesion is often photographed more than once "
  "[@cassidy]; a random 80/20 split by image would put a second photo of the same lesion into training for 38.5% of "
  "the validation images. This gives 8,583 training and 1,432 validation images. Model selection and per-site results "
  "use validation, and the official ISIC 2018 test set (1,512 images, no source label, no confirmed duplicates of "
  "training images) is scored once per final model. Each setting runs with three seeds, five for the two main models. "
  "On Fed-ISIC2019 we report the benchmark metric, the mean of per-centre balanced accuracy on the test sets, and also "
  "balanced accuracy on all test images together.")

H("B. Sites and the device-mixing control")
P("The Training images box takes HAM10000 or Fed-ISIC2019. HAM10000 has 10,015 images of 7,470 lesions in seven "
  "classes, collected from four sources: three collections from the Department of Dermatology in Vienna and one "
  "practice in Queensland [@ham]. We use the sources as sites because each one is a real clinic or imaging setup, "
  "which is what FL across hospitals would face (Table II). The sites differ nine-fold in size and also in disease "
  "mix: vidir_molemax is 94% nevi, most of them diagnosed by follow-up as in a clinic that monitors moles, and two "
  "sites have no AKIEC image at all. Fed-ISIC2019 [@flamby] has six centres (BCN, the four HAM10000 sources and MSK) "
  "with 23,247 images in eight classes.")
BANG("TABLE II. HAM10000 sources used as sites (lesion-level split). NV share and MEL (melanoma images) refer to the "
     "training images; vidir_molemax and vienna_dias have no AKIEC image.",
     [["Site", "Train", "Val.", "NV share", "MEL"],
      ["vidir_molemax (Vienna)", "3,388", "559", "94.3%", "15"],
      ["vidir_modern (Vienna)", "2,904", "466", "54.6%", "582"],
      ["rosendahl (Queensland)", "1,926", "333", "35.3%", "299"],
      ["vienna_dias (Vienna)", "365", "74", "78.6%", "58"]])
P("The images differ more than the labels (Fig. 2). Among benign nevi only, a logistic regression on six colour "
  "statistics finds the source with accuracy 0.80 (chance 0.25), and on strips of healthy skin it separates the two "
  "large Vienna sources, which used different devices, with accuracy 0.99. The label mixes differ less: the mean "
  "pairwise Jensen-Shannon divergence between sites is 0.214, close to a Dirichlet split with concentration 1.0.")
HINH("Fig. 2. Benign nevi from the four HAM10000 sources. The diagnosis is the same in every row; colour, skin tone, "
     "hair, background and dermoscope borders change with the source.")
P("The device-mixing control (mark ①) tests whether this device difference costs accuracy in FL. It builds a second "
  "set of four sites from the same training images. Each mixed site keeps nearly the same number of images per class "
  "as the real site it replaces, but draws its lesions at random from all four sources and keeps all images of a "
  "lesion together. Size, disease mix and the images themselves stay the same; only the device behind each site's "
  "images changes, so any accuracy difference between the two sets of sites is caused by devices. On Fed-ISIC2019 the "
  "centre labels of the training images are shuffled within each class in the same way.")

H("C. Preprocessing and augmentation")
P("The Preprocessing box resizes each image to 224 pixels on the short side and crops the centre (P1). A second "
  "variant, P2, adds Shades of Gray colour constancy [@sog] with p = 6, which removes part of each device's colour tint "
  "using only the image's own pixels, so it needs nothing from other sites. It lowers the accuracy of finding the "
  "source from colour statistics from 0.78 to 0.56, so it does not remove the device trace; we use it only as one of "
  "the tests in Section IV-B. The Augmentation box applies random flips and 90-degree rotations. On Fed-ISIC2019 both "
  "boxes follow the benchmark code.")

H("D. Backbone and normalization")
P("The Backbone box is where we look for the device effect, so we vary its Norm layer. A BN layer normalizes each "
  "channel with a mean and variance estimated from the training images and stores these statistics in the model; an "
  "LN layer normalizes each image with its own features and stores none. If devices shift the image statistics of a "
  "site, BN is the layer that records the shift. Each backbone keeps the Norm layer it was pretrained with: two BN "
  "models, EfficientNet-B0 [@tan] and ResNet-18 [@he]; two LN models, ConvNeXt-atto [@liu] and DeiT-tiny [@touvron]; "
  "and a ResNet-50 pair with BN and with GN. The pair shares its architecture and pretraining recipe, so a difference "
  "between its two members comes from normalization alone; to fit a 4 GB GPU it runs at 160 pixels with batch size 32.")
P("EfficientNet-B0 is the baseline because it is the model of the FLamby benchmark. ConvNeXt-atto is the main LN "
  "model because it is a convolutional network of similar size (3.4M against 4.0M weights), so it differs from the "
  "baseline in its normalization and block design rather than in scale. All backbones start from ImageNet weights, "
  "except in the proposed DP design (Section III-E). The Classifier box is one linear layer.")

H("E. Local update and aggregation")
P("In full training, the Local update box trains the whole network for one epoch on the site's images with "
  "cross-entropy weighted by inverse class frequency. The weights come from the class counts of all sites together, "
  "which requires each hospital to report its number of cases per class; weights from each site's own counts were "
  "worse (Section IV-C). The Aggregation box (②) is FedAvg (1), the most common choice and the FLamby baseline. FedAvg "
  "averages every weight, including the running BN statistics, which is the path tested in Section IV-B. FedBN "
  "[@fedbn], which keeps the BN layers at each site, is used only as one of those tests.")
P("The DP form of the Local update box gives formal privacy. Each site trains with DP-SGD [@abadi]: Poisson sampling "
  "with an expected batch of 256, per-example clipping at norm 1, Gaussian noise, and Rényi DP accounting at "
  "delta = 1e-5. Each site spends its own budget epsilon in {8, 3, 1} over all rounds, so a small site needs more "
  "noise. We apply DP at three depths: the classifier on frozen features (L0), or the last one or two blocks plus the "
  "classifier (L1, L2). The frozen part stays in evaluation mode, so BN uses fixed statistics and the gradient of "
  "each image does not depend on other images, which DP-SGD requires. The learning rate for each epsilon is chosen on "
  "validation from {16, 4, 1}. Design 2, the reference private design, is a frozen ImageNet EfficientNet-B0 with DP "
  "on the classifier.")
P("The proposed design follows the finding of [@tramer] that DP accuracy depends mostly on the quality of the frozen "
  "features. Training only the classifier keeps the number of trained weights small (8,967), so the noise added at "
  "each step does not drown the signal, and the frozen features decide the accuracy. ImageNet features were learned "
  "from everyday photos; we instead pretrain the same EfficientNet-B0 centrally for 20 epochs on the BCN training "
  "images of ISIC 2019, public dermoscopy images from a hospital outside the federation, freeze it, and let the four "
  "HAM10000 sites train only the classifier with DP-SGD (Fig. 3). It differs from Design 2 only in the frozen "
  "weights, so any change in accuracy at a given epsilon comes from the features. Before pretraining, BCN images are "
  "compared with all HAM10000 images by perceptual hash and fine-structure correlation, and confirmed duplicates are "
  "removed.")
HINH("Fig. 3. Proposed design, the DP form of the Local update box in Fig. 1. EfficientNet-B0 is pretrained on public "
     "BCN images and frozen; each hospital computes its features once and trains only the linear classifier with "
     "DP-SGD, and the server averages the classifier weights. It differs from Design 2 only in the frozen weights: "
     "BCN here, ImageNet there.")

H("F. Membership attack")
P("The Membership attack box (④) takes the view of an honest-but-curious server, which follows the protocol but "
  "tries to learn from what it receives, or of anyone who later gets the model. The attacker tries to tell whether a "
  "given image was used in training; members are training images and non-members are validation images, which come "
  "from the same distribution because the split is by lesion. The main attack is offline RMIA [@rmia], chosen because "
  "it is strong with few reference models: eight reference models per backbone, trained centrally on halves of the "
  "training set, tell the attacker how likely each image's label is when the image was not in training. A loss "
  "threshold attack [@yeom] serves as a weak baseline. We report the AUC and the true-positive rate at a 1% "
  "false-positive rate [@carlini]. To see what a server without secure aggregation [@bonawitz] learns (③), we also "
  "run one more round, with the learning rate of round 10, and attack each site's model.")

H("G. Training details and decision rule")
P("On HAM10000 all models are trained at 224 pixels with AdamW [@loshchilov], learning rate 3e-4, weight decay 1e-4, "
  "batch size 64 and mixed precision. Federated runs use 30 rounds of one local epoch with all four sites and a cosine "
  "learning-rate decay over rounds; the centralized model trains for 15 epochs with a one-cycle schedule. The Final "
  "global model box is the round with the best validation accuracy, or the best epoch for centralized runs. On "
  "Fed-ISIC2019, which has no validation split, every run is scored at the end of training. We first reproduce the "
  "FLamby protocol (EfficientNet-B0, weighted focal loss, federated SGD at 0.01, 9 rounds of 100 local steps) and then "
  "compare EfficientNet-B0 with ConvNeXt-atto under the HAM10000 recipe at 9 rounds of 100 steps and at three times "
  "the steps (27 rounds of 100), matched by 60 centralized epochs.")
P("Each choice was judged by a rule set before the runs: a change is kept only if it moves validation accuracy by more "
  "than 0.02 in the same direction in all three seeds. A difference is called consistent when it has the same sign in "
  "every seed, and within noise otherwise.")
P("FOOTNOTE:All experiments run on one laptop GPU with 4 GB of memory. A federated run on HAM10000 takes about 9 "
  "minutes with ConvNeXt-atto and 14 with EfficientNet-B0, BCN pretraining 9 minutes, a DP run at L0 a few minutes on "
  "stored features, and a long-budget run on Fed-ISIC2019 35 to 47 minutes.")

# ================================================================== IV
H("IV. Results")
P("Unless stated otherwise, numbers are validation balanced accuracy on HAM10000, averaged over three seeds.")
H("A. Device share of the FL gap")
P("Removing device differences recovers 35% to 49% of the FL gap for the three BN models and none of it for the two "
  "LN models (Table III, Fig. 4). The real and device-mixed sites (mark ① in Fig. 1) hold the same images with the "
  "same class counts, so the change comes from devices alone. For BN models, FedAvg loses 0.10 to 0.14 against "
  "centralized training, and mixing wins back about 0.05. For LN models the gap is smaller (0.076 and 0.036), and "
  "mixing does not help: ConvNeXt-atto changes by -0.004, and DeiT-tiny drops by 0.044.")
BANG("TABLE III. Device mixing recovers part of the FL gap for BN models only. Validation balanced accuracy on "
     "HAM10000 (P1, mean of three seeds). Gap = centralized minus FedAvg real; device share = (FedAvg mixed − FedAvg "
     "real) / gap; n/a: mixing lowered accuracy by more than the gap.",
     [["Backbone", "Norm", "Centralized", "FedAvg real", "FedAvg mixed", "Gap", "Device share"],
      ["EfficientNet-B0", "BN", "0.778", "0.659", "0.707", "0.118", "40%"],
      ["ResNet-18", "BN", "0.797", "0.657", "0.706", "0.140", "35%"],
      ["ResNet-50 (160 px)", "BN", "0.800", "0.698", "0.748", "0.102", "49%"],
      ["ResNet-50 (160 px)", "GN", "0.791", "0.700", "0.713", "0.092", "15%"],
      ["ConvNeXt-atto", "LN", "0.813", "0.737", "0.733", "0.076", "−5%"],
      ["DeiT-tiny", "LN", "0.779", "0.742", "0.698", "0.036", "n/a"]])
HINH("Fig. 4. Removing device differences helps batch-norm models and does not help layer-norm models. Change in "
     "validation balanced accuracy from FedAvg on the real sites to FedAvg on the device-mixed sites, mean of three "
     "seeds; the device share of each gap is given in Table III.")
P("The device effect falls on the class that matters most. For EfficientNet-B0, FL lowers melanoma recall on the test "
  "set from 0.645 to 0.380 and pushes the model toward nevi, the majority class: it labels 64% of test images as nevi, "
  "against 56% when centralized (Table IV). Device mixing brings melanoma recall back to 0.499. ConvNeXt-atto loses "
  "much less melanoma recall in FL (0.756 to 0.692). On the mixed sites its melanoma recall drops to 0.554 while its "
  "balanced accuracy stays the same, so for this LN model mixing moves errors between classes rather than removing "
  "them.")
BANG("TABLE IV. With BN, FL loses most of its melanoma recall. Official ISIC 2018 test set, mean of three seeds. "
     "MEL recall: melanoma recall; Pred. NV: share of test images labelled as nevus.",
     [["Backbone", "Training", "MEL recall", "Bal. acc.", "Pred. NV"],
      ["EfficientNet-B0", "Centralized", "0.645", "0.714", "56%"],
      ["EfficientNet-B0", "FedAvg real", "0.380", "0.601", "64%"],
      ["EfficientNet-B0", "FedAvg mixed", "0.499", "0.651", "60%"],
      ["ConvNeXt-atto", "Centralized", "0.756", "0.746", "52%"],
      ["ConvNeXt-atto", "FedAvg real", "0.692", "0.658", "51%"],
      ["ConvNeXt-atto", "FedAvg mixed", "0.554", "0.656", "61%"]])

H("B. Mechanism of the device effect")
P("Devices could hurt FL in two ways: by turning the local updates of different sites in different directions (M1), "
  "or by shifting the BN statistics that FedAvg averages (M2, mark ② in Fig. 1). Under M2, each site's BN layers learn "
  "the image statistics of its devices, and the average that FedAvg sends back matches no site. Mixing raised FedAvg "
  "accuracy by 0.047 for EfficientNet-B0 while the cosine similarity between site updates changed by at most 0.05, so "
  "M1 is rejected. Four results support M2. First, mixing helps all three BN models and neither LN model. Second, in "
  "the ResNet-50 pair, mixing gains 0.051 with BN, consistent across seeds, and 0.014 with GN, within noise. Third, "
  "FedBN, which keeps the BN statistics at each site, gains 0.080 over FedAvg for EfficientNet-B0 and 0.040 for "
  "ResNet-18 on the real sites, but only 0.013 and 0.006 on the mixed sites. Fourth, colour constancy (P2) leaves "
  "centralized accuracy unchanged and raises FedAvg by 0.024 for EfficientNet-B0 but only 0.006 for ConvNeXt-atto.")
P("Both direct fixes for BN have limits. FedBN needs the site of each image: on the test set, where the source is "
  "unknown, an ensemble of its four site models scores 0.639 against 0.659 for FedAvg. GN avoids the device effect "
  "but is weaker overall: on the mixed sites, ResNet-50 with BN beats GN by 0.035 in every seed. No design we tried "
  "reduced the rest of the gap, about 60% for BN models and nearly all of it for LN models, which remains when only "
  "disease mix and site size differ; FedProx [@fedprox] and server momentum (FedAvgM) [@hsu] did not help.")

H("C. Best federated design and the baseline")
P("An LN backbone gives the best federated model, Design 1: ConvNeXt-atto with FedAvg, 30 rounds of one local epoch, "
  "global class weights and P1. With five seeds it reaches 0.741 +/- 0.013 against 0.661 +/- 0.010 for the "
  "EfficientNet-B0 baseline (mean +/- standard deviation), with no overlap between seeds, and 0.662 against 0.597 on "
  "the test set (Table V). Centralized, the two differ by only 0.035, so most of the difference appears only in FL. "
  "Design 1 also has fewer weights than the baseline and trains faster.")
P("The learning curves show how the difference builds up (Fig. 5). The mean curve of EfficientNet-B0 levels off "
  "within ten rounds, while ConvNeXt-atto keeps improving until rounds 25 to 29. Each part of Design 1 passed the rule "
  "of Section III-G: without ImageNet pretraining accuracy falls to 0.377, with EfficientNet-B0 to 0.659, and with "
  "class weights from each site's own counts to 0.706, each lower in every seed. Two local epochs, 50 rounds, FedProx "
  "and P2 were within noise, and server momentum was much worse (0.597).")
HINH("Fig. 5. EfficientNet-B0 levels off within ten rounds while ConvNeXt-atto keeps improving. Validation balanced "
     "accuracy per round of FedAvg on the four real HAM10000 sites; lines show the mean and shaded bands the range of "
     "three seeds. Tables report the best round of each run, so their values differ slightly from these per-round "
     "means.")
BANG("TABLE V. Final designs against the baseline on HAM10000. Balanced accuracy on validation and test; five seeds "
     "for the two FedAvg models, three for the DP designs. Time per run on a 4 GB laptop GPU.",
     [["Design", "Trained weights", "Val.", "Test", "Formal privacy", "Time"],
      ["Baseline: EfficientNet-B0, FedAvg", "4.0M", "0.661", "0.597", "no", "14 min"],
      ["Design 1: ConvNeXt-atto, FedAvg", "3.4M", "0.741", "0.662", "no", "9 min"],
      ["Design 2: ImageNet features, DP at L0", "8,967", "0.430", "0.386", "epsilon = 3", "minutes"],
      ["Proposed: BCN features, DP at L0", "8,967", "0.554", "0.508", "epsilon = 3", "minutes"]])

H("D. Second dataset")
P("The device effect repeats on Fed-ISIC2019. Our run of the FLamby baseline agrees with the original within its "
  "spread: 0.624 with FedAvg, against 0.539 +/- 0.097 over five seeds in the results released with FLamby [@flamby], "
  "and the same centres are weakest. Under this protocol, device mixing raises FedAvg by 0.030, a device share of 28%, "
  "exactly at the threshold of 0.03 set in advance. With three times the training steps and the HAM10000 recipe, the "
  "gap of EfficientNet-B0 is 0.137 (0.777 to 0.640), and mixing raises FedAvg by 0.057 in every seed, a device share "
  "of 42%, close to the 40% on HAM10000. For ConvNeXt-atto, mixing adds only 0.008.")
P("The LN advantage needs the longer budget. At the benchmark budget of 9 rounds of 100 steps the two backbones are "
  "close (0.643 for EfficientNet-B0, 0.628 for ConvNeXt-atto). With three times the steps ConvNeXt-atto leads by 0.043 "
  "in the per-centre metric, consistent across seeds, because EfficientNet-B0 stays near 0.64 while ConvNeXt-atto keeps "
  "improving, as in Fig. 5. On all test images together the two tie (0.700 against 0.699): ConvNeXt-atto gains at the "
  "small centres and is slightly behind at BCN, the largest one.")

H("E. Adding privacy")
P("Federated training lowers membership leakage but does not remove it, so formal protection still needs DP. Under "
  "RMIA, the true-positive rate at a 1% false-positive rate is 0.043 for the federated EfficientNet-B0 against 0.096 "
  "when centralized, and 0.054 against 0.071 for ConvNeXt-atto (mark ④). RMIA finds four to twelve times more members "
  "than the loss attack, so these figures do not come from a weak attack. A server that sees each site's model learns "
  "more than one that sees only their average (mark ③): for EfficientNet-B0 the true-positive rate on rosendahl rises "
  "from 0.044 to 0.118, while for ConvNeXt-atto the difference is small. This is an observation on two models, and we "
  "did not test its cause; secure aggregation [@bonawitz], which shows the server only the average, would leave it "
  "only the leakage of the averaged model.")
P("DP works only on the classifier (L0). With DP at L1 and L2, accuracy falls to between 0.15 and 0.24, near chance "
  "(0.14), at every epsilon, although the same depths train well without DP. At L0 most of the cost comes from the "
  "noise: with ImageNet features, clipping alone lowers accuracy from 0.617 to 0.602, and the noise at epsilon = 3 "
  "lowers it to 0.430 (Design 2, Table VI). The best backbone for accuracy does not help here: a frozen ConvNeXt-atto "
  "stays below EfficientNet-B0 at every epsilon (0.353 against 0.430 at epsilon = 3), so the private designs keep "
  "EfficientNet-B0.")
P("Public dermoscopy features raise DP accuracy by 0.12 to 0.13 at every epsilon (Table VI, Fig. 6). At epsilon = 3 "
  "the proposed design reaches 0.554, just above the threshold of 0.55 set in advance, and without DP a linear "
  "classifier on BCN features (0.710) already beats full FedAvg training of the baseline (0.661). The gain does not "
  "come from shared images: the perceptual hash flagged 7,463 candidate pairs between BCN and HAM10000, and none was "
  "a confirmed duplicate. The trade-off narrows but remains: the proposed design is 0.19 below Design 1.")
BANG("TABLE VI. Public dermoscopy features raise DP accuracy at every epsilon. DP-SGD at L0 with a frozen "
     "EfficientNet-B0, FedAvg on the real sites; validation balanced accuracy, best learning rate per epsilon, mean of "
     "three seeds; test in parentheses.",
     [["Features", "No DP", "epsilon = 8", "epsilon = 3", "epsilon = 1"],
      ["ImageNet", "0.617 (0.565)", "0.506 (0.436)", "0.430 (0.386)", "0.325 (0.297)"],
      ["Public dermoscopy (BCN)", "0.710 (0.652)", "0.628 (0.567)", "0.554 (0.508)", "0.455 (0.403)"]])
HINH("Fig. 6. Public dermoscopy features raise DP accuracy at every privacy budget but stay below the best non-private "
     "design. DP-SGD on the classifier of a frozen EfficientNet-B0 (L0), FedAvg on the four real sites; validation "
     "balanced accuracy, best learning rate per budget, mean of three seeds. Dashed line: ConvNeXt-atto FedAvg without "
     "DP, mean of five seeds.")

# ================================================================== V, VI
H("V. Discussion and Limitations")
P("For hospitals that build a federated skin-lesion model, the normalization of the backbone mattered more than the "
  "FL algorithm. With BN, part of the accuracy lost to FL comes from device differences and falls hardest on "
  "melanoma; an LN backbone avoids that part without needing the source of each image, which FedBN needs. The rest "
  "of the gap, which comes with differences in disease mix and site size, was not reduced by any design we tested.")
P("Benchmark comparisons should state the training budget and the metric. On Fed-ISIC2019 the benchmark budget hid "
  "the LN advantage, and the advantage holds for the per-centre mean but not on all test images together.")
P("For privacy, keeping images on site is not enough. If formal protection is needed, DP at this scale works only on "
  "the classifier, and the frozen features decide how much accuracy is left: public images from another hospital give "
  "better features than ImageNet. A hospital must still choose between Design 1, more accurate with only measured "
  "protection, and the DP design, 0.19 less accurate with a formal guarantee.")
P("These conclusions have three kinds of limits.")
P("RUNIN:Scale.|The federation is small, four sites and 8,583 training images, and the smallest site has 74 "
  "validation images, so most per-site differences are not significant. The 4 GB GPU limited the ResNet-50 pair to "
  "160 pixels and ruled out larger backbones.")
P("RUNIN:Mechanism.|The four comparisons point to the BN statistics but do not prove that they are the only path. "
  "Beyond the ResNet-50 pair, LN and BN models also differ in architecture, and the LN result holds on average, not at "
  "every site; DeiT-tiny even dropped under mixing, which we cannot explain. Separating disease mix and site size from "
  "the cost of FL itself would need a control with identically distributed sites.")
P("RUNIN:Privacy.|RMIA uses reference models trained centrally while the targets are trained with FL, so the measured "
  "leakage is a lower bound. DP is counted per image; DP for whole sites is not practical with four sites. The public "
  "backbone was pretrained with one seed, BCN belongs to the same ISIC collection as HAM10000, and choosing the "
  "learning rate on validation makes validation scores slightly optimistic, so test scores are reported beside them.")
P("Future work should target the part of the gap caused by disease mix and site size, and test longer budgets and "
  "larger backbones on more federations.")
H("VI. Conclusion")
P("We measured where federated learning loses accuracy on real skin-lesion sites and what formal privacy costs. For "
  "BN models, imaging devices cause about 40% of the FL gap, through the BN statistics that the server averages; an "
  "LN backbone avoids this part and gives the best federated model, 0.741 against 0.661 for the EfficientNet-B0 "
  "baseline, and the device effect repeats on Fed-ISIC2019. Federated models still leak under a strong attack, DP "
  "works only on the last layer, and public dermoscopy features raise its accuracy at epsilon = 3 from 0.430 to 0.554. "
  "At this scale no single design wins on both sides, and the measured gap between the two is 0.19.")
H("Reproducibility")
P("All HAM10000 results use the lesion-level split with seeds 0 to 2 (0 to 4 for the main models); Fed-ISIC2019 uses "
  "the official FLamby split. Every run records its settings, code hash, data-split hash, learning curves and the "
  "expected result written before it ran. Code, data splits and run logs are available from the author on request.")

# ------------------------------------------------------------------ đánh số trích dẫn theo thứ tự xuất hiện
thu_tu = []
for t, s in M:
    for k in re.findall(r"\[@(\w+)\]", s):
        assert k in KHOA, k
        if k not in thu_tu:
            thu_tu.append(k)
moi = {k: i + 1 for i, k in enumerate(thu_tu)}
M = [[t, re.sub(r"\[@(\w+)\]", lambda m: f"[{moi[m.group(1)]}]", s)] for t, s in M]
H("References")
for k in thu_tu:
    M.append(["P", f"[{moi[k]}] {REF_CU[KHOA[k]]}"])
(D / "ref_map.json").write_text(json.dumps({str(moi[k]): str(KHOA[k]) for k in thu_tu}, indent=0))
bo = sorted(set(REF_CU) - set(KHOA.values()))
print("tài liệu bỏ (không còn trích):", bo)
chua_trich = set(KHOA) - set(thu_tu)
assert not chua_trich, chua_trich
(D / "doc_v9.txt").write_text("\n".join(f"{t}\t{s.strip()}" for t, s in M))
print("ok", len(thu_tu), "tài liệu;", sum(1 for t, s in M if s.startswith("TABLE")), "bảng;",
      sum(1 for t, s in M if s.startswith("Fig. ")), "hình")
