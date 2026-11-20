"""sua_v7.py — doc_v6.txt -> doc_v7.txt.
(1) Mục III sắp lại theo thứ tự các ô của Hình 1 (khung chung): A Accuracy, B Training images + Site split,
    C Preprocessing + Augmentation, D Backbone + Norm, E Local update + Aggregation, F Membership attack,
    G Hypotheses, H Final global model / training details.
(2) Từ vựng đơn giản hơn trên toàn bài (giữ nguyên thuật ngữ bắt buộc và mọi con số).
Bảng (dòng C) lấy nguyên từ doc_v6.txt; chỉ chú thích bảng có thể viết lại."""
import re
from pathlib import Path

D = Path(__file__).parent
CU = [l.split("\t", 1) for l in (D / "doc_v6.txt").read_text().split("\n")]
CU = [x if len(x) == 2 else [x[0], ""] for x in CU]


def bang(so, chu_thich=None):
    i = [k for k, (t, s) in enumerate(CU) if s.startswith(f"TABLE {so}.")]
    assert len(i) == 1, so
    i = i[0]
    ra = [["P", chu_thich if chu_thich else CU[i][1]]]
    k = i + 1
    while k < len(CU) and CU[k][0] == "C":
        ra.append(CU[k]); k += 1
    return ra


def tham_khao():
    i = [k for k, (t, s) in enumerate(CU) if t == "H" and s == "References"][0]
    return CU[i:]


H = lambda s: [["H", s]]
P = lambda s: [["P", s]]
HINH = lambda s: [["P", ""], ["P", s]]

M = []
M += H("Abstract")
M += P("Federated learning (FL) lets hospitals train a shared skin-lesion classifier without sharing images, but on real "
       "sites it is less accurate than centralized training, and the trained model can still reveal which images it was "
       "trained on. Studies usually report the accuracy loss as one average over sites and do not measure this leakage. We "
       "measure both on HAM10000, whose four image sources act as hospitals, and repeat the key experiments on the six "
       "centres of Fed-ISIC2019. Removing device differences between sites while keeping their images and disease mix "
       "recovers 35% to 49% of the accuracy gap of batch-norm (BN) models on HAM10000 and 28% to 42% on Fed-ISIC2019, and "
       "four controlled comparisons point to the averaged BN statistics as the reason. Mixing does not help layer-norm "
       "models, and ConvNeXt-atto gives the best federated model (0.741 balanced accuracy, against 0.661 for "
       "EfficientNet-B0), although on Fed-ISIC2019 its lead needs three times the benchmark's training steps and holds only "
       "in the per-centre metric. Under a strong membership attack (RMIA), federated models leak less than centralized "
       "ones, but a server that sees each site's BN model learns more than one that sees only their average. Differential "
       "privacy works only on the last layer; pretraining the frozen backbone on public dermoscopy images from another "
       "hospital raises its accuracy at epsilon = 3 from 0.43 to 0.55, still 0.19 below the best non-private model.")
M += P("Index Terms: federated learning, differential privacy, membership inference, batch normalization, skin lesion "
       "classification, data heterogeneity.")

# ================================================================== I
M += H("I. Introduction")
M += P("A skin-cancer classifier needs many images of every lesion type, taken with many devices. No single clinic has "
       "that much data, and patient images usually cannot leave the hospital [1]. Federated learning (FL) [2] lets "
       "hospitals train one model together while each keeps its images; only model updates go to a server.")
M += P("This setup comes with two hopes, and neither holds by default. The shared model should come close to one trained "
       "on all the data in one place, yet accuracy drops when hospitals differ in patients, disease mix and imaging "
       "devices [3], [4]. Keeping images on site should protect patients, yet a trained model can reveal whether an image "
       "was in its training set [5]-[7]. Differential privacy (DP) [8] puts a formal limit on this leakage, but it costs "
       "accuracy, and the cost grows when each hospital has little data.")
M += P("In the skin-lesion FL studies we reviewed, including the Fed-ISIC2019 benchmark of FLamby [9] and the study of "
       "Chen et al. [10], the accuracy loss is reported as one number averaged over sites, and privacy is argued from the "
       "design rather than measured. One number does not show which difference between hospitals causes the loss, and so "
       "which fix to try. Without an attack, we also do not know how much a federated model leaks or what formal "
       "protection costs on real hospitals.")
M += P("We study both questions on HAM10000 [11], whose four image sources act as hospitals, and repeat the key "
       "experiments on the six centres of Fed-ISIC2019. Each main claim rests on a comparison in which only one thing "
       "changes: a device-mixing control keeps each hospital's size and disease mix but removes device differences; a "
       "pair of ResNet-50 models differs only in the normalization layer; and on Fed-ISIC2019, two architectures are "
       "compared at the benchmark budget and with three times as many training steps. We measure leakage with RMIA [12], "
       "a strong membership attack, also for a server that sees each hospital's model before averaging, and we apply DP "
       "at three depths of the network. In short, we measure what FL costs on real hospitals in accuracy and in privacy, "
       "trace part of the accuracy cost to device differences that act through batch-normalization statistics, and "
       "reduce part of the privacy cost with public features under DP.")
M += P("This work makes four contributions.")
M += P("Section II reviews related work, Section III describes the methods, Section IV reports the results, Section V "
       "discusses their limits, and Section VI concludes.")

# ================================================================== II
M += H("II. Related Work")
M += H("A. Site differences and normalization")
M += P("FedAvg [2] averages the weights that sites send back after local training, and it loses accuracy when sites "
       "hold different data [3]. Studies usually simulate different label mixes with Dirichlet splits [3]; a feature "
       "shift, where the same class looks different at each site, comes from different scanners [13]. FedProx [4] and "
       "server momentum (FedAvgM) [3] change the optimization, and we test both. FedBN [13] keeps the batch-normalization "
       "(BN) layers at each site, which requires knowing the site of each test image; we report it with and without this "
       "knowledge. Hsieh et al. [14] showed that averaging BN statistics across sites hurts when the sites hold different "
       "data, and that group normalization (GN) [15] avoids this. Transformers with layer normalization (LN) coped better "
       "with different clients [16], and ResNets without normalization beat BN ResNets in federated training, including "
       "on Fed-ISIC2019 [17]. These studies show that BN suffers when sites differ, but not which kind of difference "
       "causes the harm. We remove only device differences, keep each hospital's disease mix fixed, and measure how much "
       "of the loss goes away for BN, GN and LN models.")
M += H("B. Skin-lesion data and benchmarks")
M += P("HAM10000 [11] collects dermoscopy images from several sources and devices and is the training set of ISIC "
       "2018 [18]. Lesions photographed more than once leak across splits made by image [19], so we split by lesion. "
       "Models learn signs of the imaging device instead of lesion features [20] and lose accuracy on new centres [21]; "
       "these results come from centralized training, and we measure what such differences cost in FL. Colour constancy "
       "methods such as Shades of Gray [22] reduce the colour tint of each device, and we test one as a fix in FL. "
       "FLamby [9] released Fed-ISIC2019, which splits ISIC 2019 into six centres by imaging system and trains for a "
       "fixed, short budget. Later runs with longer training [23] or with the checkpoint chosen on validation [24] report "
       "different FedAvg accuracy; we reproduce the original protocol and add longer budgets. Chen et al. [10] build "
       "clients from real skin-lesion sources and report accuracy; we report per-site accuracy and membership leakage of "
       "the same models.")
M += H("C. Membership inference")
M += P("Membership inference asks whether a record was in a model's training data [5]. The loss threshold attack is "
       "closely tied to overfitting [6], and Nasr et al. [7] studied membership inference in FL with access to model "
       "updates. Carlini et al. [25] argued for reporting the true-positive rate at a low false-positive rate and proposed "
       "LiRA, which needs many shadow models; RMIA [12] reaches similar power with a few reference models. We use offline "
       "RMIA with eight reference models and report this rate at a 1% false-positive rate together with the AUC. The risk "
       "can differ between groups even when the average looks low [26], so we also compute it per site and per class. "
       "Secure aggregation [27] hides the update of each site from the server; we measure what it hides by attacking each "
       "site model and their average.")
M += H("D. Differential privacy")
M += P("DP-SGD [8] clips the gradient of each example and adds Gaussian noise at every step. Tramèr and Boneh [28] "
       "showed that under DP, a linear classifier on good fixed features is a strong baseline, and that better features "
       "help more than a better private optimizer. We apply this idea to federated training on medical images, using "
       "features learned from public dermoscopy images of another hospital, and compare them with ImageNet features at "
       "the same budget. Ziller et al. [29] measured the cost of DP on medical images, including HAM10000, in centralized "
       "training; we measure it across four real sites, each spending its own budget, at three depths. The cost of DP "
       "falls most on small groups [30]; we test whether this holds for small hospitals.")

# ================================================================== III (theo thứ tự các ô của Hình 1)
M += H("III. Methodology")
M += P("Fig. 1 shows the pipeline that all experiments share. A box with a general name, such as Backbone or "
       "Aggregation, has several variants, and the section printed in its corner describes them; the subsections below "
       "follow the boxes in the same order. A box with a specific label, such as the linear Classifier, is the same in "
       "every experiment. The numbered marks show where the study changes or measures something: the site split (①), "
       "the averaging at the server, which also averages the BN statistics (②), the site models that the server sees "
       "(③), and the attacks on the final model (④).")
M += HINH("Fig. 1. Pipeline shared by all experiments. Each of the K hospitals trains the same network on its own "
          "images, and the server combines the site models into the global model of the next round. A box with a "
          "general name has several variants, described in the section printed in its corner. Marks: ① site split, "
          "real or device-mixed; ② averaging at the server, including the BN statistics, the path tested in Section "
          "IV-B; ③ site models seen by a server without secure aggregation (Section IV-D); ④ membership attacks on the "
          "final model.")

M += H("A. Setup and metrics")
M += P("Each image has one of C = 7 labels. The training data are held by K = 4 sites, and site k holds n_k of the n "
       "images. In each round the server sends the global weights to every site, each site trains on its own images, "
       "and the server averages the results in proportion to site size:")
M += P("Always predicting nevus gives about 67% accuracy, so the Accuracy box uses balanced accuracy, the mean recall "
       "over classes, which is also the ISIC 2018 metric [18]:")
M += P("The cost of FL is the gap to a centralized model with the same backbone and recipe. To split this gap, each "
       "design is also trained on device-mixed sites (Section III-B). The device share is the accuracy that FedAvg gains "
       "when moving from the real to the device-mixed sites, divided by the gap. Both sets of sites hold the same images "
       "with the same class counts per site, so the device share is the part of the gap that goes away when only device "
       "differences are removed.")
M += P("Training and validation images are split by lesion; a split by image would put a second photo of the same "
       "lesion into training for 38% of the validation images. This gives 8,583 training and 1,432 validation images. "
       "The official ISIC 2018 test set (1,512 images) is scored once per final model, and a check with perceptual "
       "hashes found no confirmed duplicates with the training data. The test set has no source label, so per-site "
       "results and model selection use the validation split. Repeated seeds (three, or five for the main settings) show "
       "whether a result is stable across runs, and a bootstrap over lesions shows how much a per-site or per-class "
       "result depends on the particular images. On Fed-ISIC2019 we report the benchmark metric, the mean of per-centre "
       "balanced accuracy on the test sets, and also balanced accuracy on all test images together (pooled).")

M += H("B. Training images and site split")
M += P("The Training images box takes HAM10000 or Fed-ISIC2019, and the Site split box (①) divides the images into "
       "sites, either by their real source or with the device-mixing control. HAM10000 has 10,015 images of 7,470 "
       "lesions in seven classes (Table I). Its four sources, three collections from the Department of Dermatology in "
       "Vienna and one practice in Queensland [11], serve as the sites (Table II). They differ nine-fold in size, and 94% "
       "of the labels at vidir_molemax come from follow-up rather than histopathology, as in a clinic that monitors "
       "moles. Fed-ISIC2019 [9] has six centres (BCN, the four HAM10000 sources and MSK) with 23,247 images in eight "
       "classes in its official split.")
M += bang("I")
M += bang("II")
M += P("The sources differ little in labels but a lot in images. The mean pairwise Jensen-Shannon divergence of their "
       "class mixes is 0.214, close to a Dirichlet split with concentration 1.0 (0.213). The images differ much more "
       "(Fig. 2). A logistic regression on six colour statistics of nevi finds the source with accuracy 0.80 (chance "
       "0.25, diagnosis alone 0.42), and on strips of healthy skin it separates the two large Vienna sources, which used "
       "different devices, with accuracy 0.99. The dark corners of some dermoscope images do not explain this, and Shades "
       "of Gray removes only part of the device trace (0.78 to 0.56 across the four sources).")
M += HINH("Fig. 2. Benign nevi from the four HAM10000 sources. The diagnosis is the same in every row; colour, skin "
          "tone, hair, background and dermoscope borders change with the source.")
M += P("The device-mixing control builds a second set of sites from the same training images. Each mixed site keeps "
       "nearly the same number of images per class as the real site it replaces, but draws its lesions at random from "
       "all four sources and keeps all images of a lesion together. The two sets of sites therefore differ only in which "
       "device each site's images come from; size, disease mix and the images themselves stay the same. On Fed-ISIC2019 "
       "the centre labels of the training images are shuffled within each class, which keeps the class counts of every "
       "client.")

M += H("C. Preprocessing and augmentation")
M += P("The Preprocessing box has two variants on HAM10000. P1 resizes each image to 224 pixels on the short side and "
       "crops the centre. P2 adds Shades of Gray colour constancy [22] with p = 6, which uses only the pixels of each "
       "image and so needs nothing from other sites. The Augmentation box applies random flips and 90-degree rotations on "
       "HAM10000. On Fed-ISIC2019 both boxes follow the benchmark code.")

M += H("D. Backbone and normalization")
M += P("The Backbone box stands for six networks, each kept with the Norm layer it was pretrained with: two BN models "
       "(EfficientNet-B0 [31], ResNet-18 [32]), two LN models (ConvNeXt-atto [33], DeiT-tiny [34]), and a ResNet-50 pair "
       "with BN and with GN. The pair shares its architecture and pretraining recipe and differs only in the Norm layer, "
       "so a difference between them comes from normalization. To fit a 4 GB GPU, the pair runs at 160 pixels with batch "
       "size 32. All backbones start from ImageNet weights, except in the proposed design, which uses weights learned "
       "from public dermoscopy images (Section III-E). The Classifier box is always one linear layer.")

M += H("E. Local update and aggregation")
M += P("The Local update box has two forms. In full training, each site trains the whole network with class-weighted "
       "cross-entropy. Class weights are inverse class frequencies, computed either from the counts of all sites "
       "together (global), which requires each hospital to report its number of cases per class, or from each site's "
       "own counts (local). The Aggregation box (②) is FedAvg in the reference design, compared with FedProx [4] "
       "(mu = 0.01, 0.1 and 1), FedAvgM (beta = 0.9) and FedBN [13]. FedAvg averages all weights, including the running "
       "statistics of the BN layers. FedBN keeps the BN layers at each site; it is scored with each site's own model on "
       "that site's validation images and, on the test set, where the source is unknown, with an ensemble of the four "
       "site models.")
M += P("In the DP form, each site trains with DP-SGD [8]: Poisson sampling, per-example clipping at norm 1, Gaussian "
       "noise and Rényi DP accounting at delta = 1e-5, with each site spending its own budget epsilon in {8, 3, 1} over "
       "all rounds. DP is applied to the classifier on frozen features (L0) or to the last one or two blocks plus the "
       "classifier (L1, L2). The frozen part stays in evaluation mode, so BN uses fixed statistics and the per-example "
       "gradients stay valid. A run with clipping only and a run without DP separate the cost of clipping from the cost "
       "of noise. The learning rate for each epsilon is chosen on validation from {16, 4, 1}.")
M += P("The proposed design follows the finding of [28] that DP accuracy depends mostly on the quality of the frozen "
       "features. We pretrain the same backbone centrally on public dermoscopy images from another hospital, the BCN "
       "training images of ISIC 2019, freeze it, and let the four HAM10000 sites train only the classifier with DP-SGD "
       "(Fig. 3). It differs from the ImageNet version only in the frozen weights, so any change in accuracy at a given "
       "epsilon comes from the features. Before pretraining, BCN images are compared with all HAM10000 images by "
       "perceptual hash and fine-structure correlation, and confirmed duplicates are removed, so no HAM10000 image enters "
       "the public data.")
M += HINH("Fig. 3. Proposed design, the DP form of the Local update box in Fig. 1. EfficientNet-B0 is pretrained on "
          "public BCN images and frozen; each hospital computes its features once and trains only the linear classifier "
          "with DP-SGD, and the server averages the classifier weights. It differs from Design 2 only in the frozen "
          "weights: BCN here, ImageNet there.")

M += H("F. Membership attacks")
M += P("The Membership attack box (④) takes the view of an honest-but-curious server, which follows the protocol but "
       "tries to learn from what it receives, or of anyone who later gets the global model. The attacker tries to tell "
       "whether a given image was used in training. Members are training images and non-members are validation images; "
       "both come from the same distribution because the split is by lesion. The loss threshold attack [6] is a weak "
       "baseline, and offline RMIA [12] is the main attack. For each backbone, eight reference models are trained "
       "centrally on halves of the training set, split by lesion into four pairs of halves, so each training image is "
       "unseen by four reference models and each validation image by all eight. RMIA divides the target model's "
       "probability of the true label by a reference term computed from the models that did not see the image; the "
       "weight of this term is tuned with a reference model acting as the target, never with the real target. We report "
       "the AUC of the attack and the true-positive rate at a 1% false-positive rate [25].")
M += P("Without secure aggregation [27], the server sees each site's model after local training (③). We simulate one "
       "extra round starting from the best global model of each FedAvg run. The cosine schedule makes the learning rate "
       "of that late round very small, so local training uses the learning rate of round 10, when updates have their "
       "usual size. We attack each site's own model, the averaged model and the global model before the round, with the "
       "site's training images as members and its validation images as non-members. The three models differ only in "
       "what the server sees, so the comparison measures what secure aggregation hides.")

M += H("G. Hypotheses and decision rules")
M += P("Each run was logged with its expected result before it started. A design replaces the reference only if it is "
       "better on validation in all three seeds and by more than 0.02 on average. A part of a final design is kept only "
       "if removing it lowers validation accuracy by more than 0.02 in all three seeds. The device effect counts as "
       "repeated on Fed-ISIC2019 if mixed minus source is above 0.03 in every seed. Differences between sites or classes "
       "are claimed only when bootstrap intervals do not overlap. Five hypotheses are tested: M1, M2, A, H1 and H2.")

M += H("H. Training details")
M += P("On HAM10000 all models are trained at 224 pixels with AdamW [35], learning rate 3e-4 and weight decay 1e-4, "
       "batch size 64, class-weighted cross-entropy and mixed precision. The centralized model trains for 15 epochs with "
       "a one-cycle schedule. Federated runs use 30 rounds of one local epoch with all four sites and a cosine "
       "learning-rate decay over rounds. The Final global model box is the round with the best validation accuracy, or "
       "the best epoch for centralized runs; on Fed-ISIC2019, which has no validation split, every run is scored at the "
       "end of training. DP runs at L0 train the classifier for 30 rounds with an expected batch of 256.")
M += P("On Fed-ISIC2019 we first follow the FLamby protocol: EfficientNet-B0, weighted focal loss, centralized Adam at "
       "5e-4 for 20 epochs, and federated SGD at 0.01 with 9 rounds of 100 local steps, scoring the final model. To "
       "compare architectures we then train EfficientNet-B0 and ConvNeXt-atto with the HAM10000 recipe under three "
       "federated budgets: 9 rounds of 100 steps; the same steps over 27 rounds, which changes only how often the server "
       "averages; and three times the steps (27 rounds of 100), matched by 60 centralized epochs.")
M += P("FOOTNOTE:All experiments run on one laptop GPU with 4 GB of memory. A federated run on HAM10000 takes about 9 "
       "minutes with ConvNeXt-atto and 14 with EfficientNet-B0, a DP run at L0 a few minutes on stored features, and a "
       "long-budget run on Fed-ISIC2019 about 35 minutes. ResNet-50 at 224 pixels needed 6.8 GB, hence the 160-pixel "
       "pair. Runs resume from checkpoints and log their settings, code hash, data-split hash and learning curves.")

# ================================================================== IV
M += H("IV. Results")
M += P("Unless stated otherwise, numbers are validation balanced accuracy on HAM10000, averaged over three seeds. A "
       "difference is called consistent when it has the same sign in every seed, and within noise otherwise; claims "
       "about single sites or classes rest on bootstrap intervals.")
M += H("A. Device share of the federation gap")
M += P("The real and device-mixed sites hold the same images with the same class counts per site, so the change in "
       "accuracy between them comes from devices (Table III, Fig. 4). For the three BN models, FedAvg loses 0.10 to 0.14 "
       "against centralized training, and device mixing wins back about 0.05 of it, a device share of 35% to 49%. For "
       "the two LN models the gap is smaller (0.076 and 0.036) and mixing does not help: ConvNeXt-atto changes by "
       "-0.004, and DeiT-tiny drops by 0.044 and stays below the real sites in all three mixing draws. Devices thus "
       "explain a large part of the gap for BN models and no measurable part for LN models.")
M += bang("III", "TABLE III. Device mixing recovers part of the federation gap for BN models only. Validation balanced "
                 "accuracy on HAM10000 (P1, mean of three seeds). Gap = central minus FedAvg source; device share = "
                 "(mixed − source) / gap; n/a: mixing lowered accuracy by more than the gap. Mixed values use one draw "
                 "of the device-mixed sites; for DeiT-tiny two more draws gave 0.718 and 0.718, also below the real "
                 "sites.")
M += HINH("Fig. 4. Removing device differences helps batch-norm models and does not help layer-norm models. Change in "
          "validation balanced accuracy from FedAvg on the real sites to FedAvg on the device-mixed sites, mean of "
          "three seeds; the device share of each gap is given in Table III.")
M += P("The smaller gap also gives a better federated model. With five seeds, FedAvg reaches 0.741 +/- 0.013 with "
       "ConvNeXt-atto and 0.661 +/- 0.010 with EfficientNet-B0 (mean +/- standard deviation), and the seed ranges do "
       "not overlap (0.719 to 0.752 against 0.646 to 0.672); on the test set the two score 0.662 and 0.597. "
       "Centralized, ConvNeXt-atto is ahead by only 0.035 (Table III), so most of the difference appears only in FL.")
M += H("B. Mechanism of the device effect")
M += P("Four results point to the BN statistics that FedAvg averages (M2, mark ② in Fig. 1), and none to the "
       "direction of local updates (M1). Mixing raised FedAvg accuracy by 0.047 for EfficientNet-B0 while the cosine "
       "similarity between site updates changed by at most 0.05, so M1 is rejected. For M2, first, mixing helps all "
       "three BN models and neither LN model. Second, in the ResNet-50 pair, mixing gains 0.051 with BN, consistent "
       "across seeds, and 0.014 with GN, within noise. Third, FedBN, which keeps the BN statistics at each site, gains "
       "0.080 over FedAvg for EfficientNet-B0 and 0.040 for ResNet-18 on the real sites, but only 0.013 and 0.006 on "
       "the mixed sites. Fourth, colour constancy (P2) leaves centralized accuracy unchanged and raises FedAvg by 0.024 "
       "for EfficientNet-B0 but only 0.006 for ConvNeXt-atto.")
M += P("Both direct fixes have practical limits. FedBN needs the site of each image: with an ensemble of the four site "
       "models, as on the test set, it scores 0.639 against 0.659 for FedAvg. GN avoids the device effect but is weaker "
       "overall: on the mixed sites, ResNet-50 with BN beats GN by 0.035 in every seed.")
M += P("The rest of the gap, about 60% for BN models and nearly all of it for LN models, remains when only disease mix "
       "and site size differ, and no design we tried reduced it. FedProx with mu = 1 cut the drift of the largest site "
       "by a factor of three without improving accuracy (+0.009), and FedProx and FedAvgM brought no gain for "
       "ConvNeXt-atto (FedAvgM fell to 0.597). Per site, only rosendahl has a gap whose bootstrap interval excludes zero "
       "([0.04, 0.21]); the point gap of vienna_dias (0.31) rests on 74 validation images and is not significant.")
M += H("C. Ablation")
M += P("Each final design is tested by changing one part at a time, with the rule of Section III-G (Table IV). Design "
       "1 is ConvNeXt-atto with FedAvg, 30 rounds of one local epoch, global class weights and P1. ImageNet "
       "pretraining, the ConvNeXt-atto backbone and global class weights pass the rule, and server momentum (FedAvgM) is "
       "clearly worse than FedAvg; the other changes are within noise. Global class weights therefore stay, at the cost "
       "that each hospital reports its number of cases per class.")
M += P("Design 2 is a frozen EfficientNet-B0 with DP-SGD on the classifier (L0), 30 rounds and P1; its learning rate "
       "is chosen on validation for each epsilon. At epsilon = 3, 30 rounds and the EfficientNet-B0 backbone pass the "
       "rule against 10 rounds and ConvNeXt-atto, while 60 rounds, P2 and a ViT-S backbone are within noise. The "
       "proposed design (Section IV-E) is Design 2 with the ImageNet weights of the frozen backbone replaced by weights "
       "pretrained on public BCN images.")
M += bang("IV", "TABLE IV. Ablation of the two final designs. Each row changes one part and keeps the rest of the design. "
                "Validation balanced accuracy, mean of three seeds; Diff. = change from the design; rule met = lower by "
                "more than 0.02 in every seed (Section III-G).")
M += H("D. Membership leakage")
M += P("A low leakage figure can mean a private model or a weak attack, so Table V reports a weak and a strong attack. "
       "RMIA beats the loss attack for every model and finds four to twelve times more members at a 1% false-positive "
       "rate. Even under RMIA, federated models leak less than centralized ones (true-positive rate 0.043 against 0.096 "
       "for EfficientNet-B0, 0.054 against 0.071 for ConvNeXt-atto), and the two federated models leak about equally.")
M += bang("V")
M += P("Leakage differs more between classes than between sites. For the centralized EfficientNet-B0 under the loss "
       "attack, AKIEC reaches an AUC of 0.872 and nevi 0.536 (AKIEC is above 0.84 and nevi below 0.55 in every seed), "
       "while across sites the AUC ranges only from 0.525 at vidir_molemax to 0.700 at rosendahl, and the smallest site, "
       "vienna_dias (0.623), leaks less than rosendahl and vidir_modern. H2 therefore holds for classes and is rejected "
       "for sites. Across the seven classes, leakage goes with the train-validation loss gap (Spearman 0.75) more than "
       "with class size (-0.46): it reflects memorization rather than rarity, and VASC is rare but leaks little.")
M += P("A curious server that sees each site's model before averaging learns more than one that sees only the average "
       "(mark ③ in Fig. 1). For EfficientNet-B0, the rosendahl model gives an RMIA AUC of 0.672 and a true-positive rate "
       "of 0.118, against 0.577 and 0.044 for the averaged model; for vidir_modern the AUC rises from 0.622 to 0.676. "
       "With the smaller learning rate of the best round the direction is the same and the differences are smaller "
       "(rosendahl 0.585 to 0.658). For ConvNeXt-atto the difference is small or absent. For EfficientNet-B0 the "
       "averaged model leaks about as much as the global model before the round, so with secure aggregation the server "
       "learns little more than anyone who holds the final model.")
M += H("E. Cost of differential privacy")
M += P("Clipping alone costs little; most of the cost of DP comes from the noise. With ImageNet features at L0, "
       "clipping lowers accuracy from 0.617 to 0.602, and the noise at epsilon = 3 lowers it further to 0.430 (Table "
       "VI). Training deeper layers helps without DP (the last two blocks of ConvNeXt-atto reach 0.702), but with DP at "
       "L1 and L2 accuracy falls to between 0.15 and 0.24, near chance (0.14), at every epsilon, also with a batch of "
       "1024 and a learning rate of 3e-4. Formal privacy is thus possible only at L0. There, ConvNeXt-atto stays below "
       "EfficientNet-B0 at every epsilon (0.409, 0.353 and 0.276 at epsilon = 8, 3 and 1), so hypothesis A is rejected; "
       "among eight frozen ImageNet backbones, EfficientNet-B0 is the best at epsilon = 3.")
M += bang("VI")
M += HINH("Fig. 5. Public dermoscopy features raise DP accuracy at every privacy budget but stay below the best "
          "non-private design. DP-SGD on the classifier of a frozen EfficientNet-B0 (L0), FedAvg on the four real "
          "sites; validation balanced accuracy, best learning rate per budget, mean of three seeds. Dashed line: "
          "ConvNeXt-atto FedAvg without DP, mean of five seeds.")
M += P("With backbone weights pretrained on public BCN images instead of ImageNet, accuracy rises by 0.12 to 0.13 at "
       "every epsilon (Fig. 5). Without DP, the classifier on BCN features (0.710) already beats full FedAvg training "
       "of EfficientNet-B0 (0.661). At epsilon = 3 it reaches 0.554, above the threshold of 0.55 set in advance by less "
       "than the spread between seeds, so we read it as a clear gain over ImageNet features but not a reliable pass of "
       "the threshold. It stays 0.19 below Design 1: the trade-off narrows but does not disappear. The gain does not "
       "come from shared images: the perceptual hash flagged 7,463 candidate pairs between BCN and HAM10000, and none "
       "was a confirmed duplicate.")
M += P("The cost of DP is uneven. At epsilon = 3, vienna_dias needs a noise multiplier of 8.2 to spend its budget and "
       "vidir_molemax only 2.5, yet vidir_molemax loses the most accuracy and vienna_dias the least, so the site part "
       "of H1 is rejected. Among classes, dermatofibroma loses the most, while melanoma, already low without DP, does "
       "not. Centralized DP at L0 is about 0.04 better than federated DP.")
M += H("F. Second dataset and training budget")
M += P("Our run of the FLamby protocol on Fed-ISIC2019 agrees with the original results within their spread. We "
       "obtain 0.731 centralized, 0.624 with FedAvg and 0.652 with FedProx, against 0.708 +/- 0.016, 0.539 +/- 0.097 "
       "and 0.544 +/- 0.074 (mean +/- standard deviation over five seeds) in the results released with FLamby [9]. The "
       "per-centre pattern also repeats: rosendahl is the weakest centre even when centralized (0.630; FLamby 0.623), "
       "and MSK drops most under FedAvg (0.415; FLamby 0.387). Under this budget, device mixing raises FedAvg by 0.030 "
       "(device share 28%), exactly at the threshold for a repeated effect.")
M += P("The ranking of the two architectures depends on the budget (Table VII). With 9 rounds of 100 steps they tie, "
       "and spreading the same steps over 27 rounds changes nothing. With three times the steps, ConvNeXt-atto leads by "
       "0.043 in the per-centre metric, consistent across seeds. On all test images together the two tie (0.700 against "
       "0.699), because ConvNeXt-atto gains most at the smaller centres and is slightly behind at BCN, the largest one. "
       "EfficientNet-B0 stays near 0.64 under every budget while ConvNeXt-atto keeps improving, as on HAM10000 (Fig. "
       "6), where the mean curve of EfficientNet-B0 levels off within ten rounds and ConvNeXt-atto improves until "
       "rounds 25 to 29.")
M += bang("VII")
M += HINH("Fig. 6. EfficientNet-B0 levels off within ten rounds while ConvNeXt-atto keeps improving. Validation "
          "balanced accuracy per round of FedAvg on the four real HAM10000 sites; lines show the mean and shaded bands "
          "the range of three seeds. Tables report the best round of each run, so their values differ slightly from "
          "these per-round means.")
M += P("With the long budget and a matched 60-epoch centralized run, the device effect repeats at a similar size. The "
       "gap to centralized training is 0.137 for EfficientNet-B0 (0.777 to 0.640) and 0.053 for ConvNeXt-atto (0.736 "
       "to 0.683). Device mixing raises FedAvg by 0.057 for EfficientNet-B0, consistent across seeds, a device share of "
       "42% (40% on HAM10000), and by 0.008 for ConvNeXt-atto. On all test images together the gains are 0.048 and "
       "0.030, so mixing helps EfficientNet-B0 more under both metrics but still changes ConvNeXt-atto. Centralized, "
       "EfficientNet-B0 is stronger (0.777 against 0.736), mostly because of one centre, vidir_modern (0.858 against "
       "0.681); in FL, ConvNeXt-atto is stronger in the per-centre metric only. Per centre, mixing still helps "
       "ConvNeXt-atto at MSK (0.46 to 0.65) and rosendahl (0.53 to 0.61) and hurts it at vidir_modern and vienna_dias, "
       "so for LN the device effect is small on average but not absent.")

# ================================================================== V, VI
M += H("V. Discussion and Limitations")
M += P("On real hospitals, about 40% of the FL gap for BN models comes from imaging devices, and four comparisons point "
       "to the averaged BN statistics as the reason. An LN backbone removes that part without needing the source of "
       "each image, which FedBN needs, and on Fed-ISIC2019, with a long enough budget, mixing again helps the BN model "
       "more than the LN model. The rest of the gap remains when only disease mix and site size differ, and no design we "
       "tested reduced it.")
M += P("The Fed-ISIC2019 results also hold a lesson on method. Under the FLamby budget the two architectures tie; "
       "with three times the steps they differ by 0.043, because EfficientNet-B0 levels off early while ConvNeXt-atto "
       "keeps learning. A short benchmark budget can therefore reverse a comparison between architectures in FL. Had we "
       "stopped at the first budget, we would have concluded that the LN advantage was specific to HAM10000. The "
       "comparison also depends on the metric: the 0.043 lead holds for the per-centre mean that FLamby reports, while "
       "on all test images together the two models tie, so comparisons on this benchmark should name the metric and "
       "report both.")
M += P("On privacy, federated training lowers membership leakage but does not remove it, and a server that sees "
       "individual site models learns more than one that sees only their average, so secure aggregation is worth its "
       "cost, at least for BN models. Without DP, the choice of backbone barely changes leakage. At this scale, formal "
       "guarantees are possible only at the last layer, where public dermoscopy features raise accuracy by 0.12 to 0.13 "
       "at every epsilon. The design narrows the trade-off in our setting but does not remove it: at epsilon = 3 it "
       "stays 0.19 below Design 1, which itself offers only measured, not formal, protection.")
M += P("These conclusions have limits. The federation is small, four sites and 8,583 training images, and the "
       "smallest site has 74 validation images, so most per-site differences are not significant. RMIA uses reference "
       "models trained centrally while the targets are trained with FL, so the measured leakage is a lower bound. The "
       "device share depends on how the mixed sites are built; the LN result holds on average, not at every site, and "
       "the drop of DeiT-tiny under mixing is unexplained. The rest of the gap mixes differences in disease mix and "
       "size with the cost of FL itself; a control with identically distributed sites would be needed to separate them. "
       "Beyond the ResNet-50 pair, LN and BN models also differ in architecture, so part of the LN advantage may come "
       "from the architecture itself. We did not study why EfficientNet-B0 is stronger centrally on Fed-ISIC2019, an "
       "advantage concentrated in one centre. DP is counted per image, with each site spending its own budget; DP at "
       "the level of whole sites is not practical with four sites. The public backbone was pretrained with one seed, "
       "BCN belongs to the same ISIC collection as HAM10000, and choosing the learning rate on validation makes "
       "validation scores slightly optimistic, so test scores are reported beside them. Source inference, gradient "
       "inversion and malicious sites are out of scope, and the 4 GB GPU limited the ResNet-50 pair to 160 pixels and "
       "ruled out larger backbones.")
M += P("Future work should target the part of the gap caused by disease mix and site size, test longer budgets and "
       "larger backbones on more federations, and measure whether a federated model reveals a patient's hospital beyond "
       "what the image already shows.")
M += H("VI. Conclusion")
M += P("We measured what federated learning costs on real hospitals in accuracy and in privacy, on the four sources "
       "of HAM10000 and the six centres of Fed-ISIC2019. For BN models, imaging devices cause about 40% of the accuracy "
       "gap, and four comparisons point to the averaged BN statistics as the reason; an LN backbone avoids this part, "
       "while its advantage on Fed-ISIC2019 depends on the training budget and the metric. Federated training leaks less "
       "than centralized training, individual site models leak more than their average, and DP works only at the last "
       "layer, where public dermoscopy features recover part of its cost. At this scale no single design wins on both "
       "sides: Design 1 reaches 0.741 without formal guarantees, and the proposed DP design reaches 0.554 at epsilon = 3.")
M += H("Reproducibility")
M += P("All HAM10000 results use the lesion-level split with seeds 0 to 2 (0 to 4 for the main settings); "
       "Fed-ISIC2019 uses the official FLamby split. Every run records its settings, code hash, data-split hash, "
       "learning curves and the expected result written before it ran. Code, data splits and run logs are available "
       "from the author on request.")
M += tham_khao()

# ------------------------------------------------------------------ kiểm: mọi con số của doc_v6 (ngoài tham khảo) còn đủ
def so(txt):
    return re.findall(r"\d+(?:[.,]\d+)*%?", txt)
cu_txt = " ".join(s for t, s in CU[:[k for k, (t, s) in enumerate(CU) if s == "References"][0]])
moi_txt = " ".join(s for t, s in M[:[k for k, (t, s) in enumerate(M) if s == "References"][0]])
from collections import Counter
thieu = Counter(so(cu_txt)) - Counter(so(moi_txt))
print("số có trong v6 mà v7 thiếu:", dict(thieu))
(D / "doc_v7.txt").write_text("\n".join(f"{t}\t{s.strip()}" for t, s in M))
print("ok", len(M), "dòng")
