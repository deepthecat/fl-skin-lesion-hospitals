"""viet_v10.py — bản gọn (11.2026): article hai cột, đề mục đậm đánh số 1, 1.1; tóm tắt chữ thường; chú thích chữ thường;
không chữ máy, không ký hiệu khoanh tròn; ít trích dẫn, ít nhãn; chỉ mạch chính (hiệu năng) và một mạch phụ (riêng tư).
Số liệu lấy từ doc_v9 (đã đối chiếu JSON/CSV) và recall_theo_lop.csv. Chạy: python viet_v10.py && xelatex paper.tex (2 lần)."""
import re
import shutil
from pathlib import Path

D = Path(__file__).parent
FIG = D / "figures"
FIG.mkdir(exist_ok=True)
for src, dst in (("/home/claude/hinh_moi/v4/hinh_paper/hinh1_pipeline.pdf", "fig1_pipeline.pdf"),
                 ("/home/claude/hinh_moi/v4/hinh_paper/hinh2_nevi.pdf", "fig2_nevi.pdf"),
                 ("/home/claude/hinh_moi/v4/hinh_paper/hinh3_thiet_ke.pdf", "fig3_design.pdf"),
                 ("/home/claude/hinh_moi/v4b/hinh_paper/hinh5_rieng_tu.pdf", "fig4_privacy.pdf")):
    shutil.copy(src, FIG / dst)

# ------------------------------------------------------------------ tài liệu (nguyên văn từ danh sách cũ đã kiểm)
CU = {}
for l in (D / "doc_v8.txt").read_text().split("\n"):
    m = re.match(r"^P\t\[(\d+)\]\s*(.*)$", l)
    if m:
        CU[int(m.group(1))] = m.group(2)
KHOA = dict(mcmahan=2, abadi=8, flamby=9, ham=11, hsieh=14, fedbn=13, tramer=28, rmia=12, chen=10, tan=31, liu=33)
LINK = {2: "https://proceedings.mlr.press/v54/mcmahan17a.html",
        9: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract.html",
        12: "https://proceedings.mlr.press/v235/zarifzadeh24a.html",
        13: "https://openreview.net/pdf/37e144ecec7a76ce5093d4ec328649e8ef208a5b.pdf",
        14: "https://proceedings.mlr.press/v119/hsieh20a.html",
        28: "https://openreview.net/forum?id=YTWGvpFOQD-",
        31: "https://proceedings.mlr.press/v97/tan19a.html",
        33: "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html"}


def esc(s):
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def tai_lieu(n_cu):
    s = CU[n_cu]
    s = re.sub(r"pp\. (\d+)-(\d+)", r"pp. \1@@\2", s)
    doi = None
    m = re.search(r", doi: (\S+?)\.?$", s)
    if m:
        doi, s = m.group(1), s[:m.start()] + "."
    dau, duoi = s.split('," ', 1)
    mo = "in " if duoi.startswith("in ") else ""
    duoi = duoi[len(mo):]
    m = re.match(r"^(.*?)(, (?:vol\.|PMLR|no\.|pp\.|art\.|Datasets|\d{4}).*)$", duoi)
    s = esc(dau + '," ') + mo + r"\textit{" + esc(m.group(1)) + "}" + esc(m.group(2))
    s = s.replace("@@", "--")
    if doi:
        s = s.rstrip(".") + r", doi: \href{https://doi.org/" + doi + "}{" + esc(doi) + "}."
    if n_cu in LINK:
        s = s.rstrip(".") + r". Available: \url{" + LINK[n_cu] + "}"
    return s


# ------------------------------------------------------------------ nội dung
TEX = r"""
\section{Introduction}
A skin-cancer classifier needs many images of every lesion type, taken with many devices. No single clinic has that much data, and patient images usually cannot leave the hospital. Federated learning, or FL, lets hospitals train one model together [@mcmahan]: each site trains on its own images, and a server averages the site models into one global model.

This comes at a price on two sides. Hospitals differ in patients, disease mix and imaging devices, and the averaged model is less accurate than one trained on all images in one place. Keeping images on site is also not a formal guarantee, because a trained model can reveal whether an image was in its training set. Differential privacy, or DP, gives such a guarantee, at a further cost in accuracy [@abadi].

Studies of FL on skin images, such as the FLamby benchmark [@flamby], report the accuracy loss as one number averaged over sites. One number does not tell which difference between hospitals causes the loss, and so which fix to try. Privacy is usually argued from the design rather than measured.

We trace the accuracy loss on HAM10000 [@ham], whose four image sources act as hospitals. A device-mixing control rebuilds the sites from the same images with the same disease mix but with devices mixed across sites, so any change in accuracy comes from devices alone. Comparing backbones with different normalization layers then shows where the device effect enters the model, and leads to the best federated design. We repeat the key test on Fed-ISIC2019 and then add privacy to the design. This work makes three contributions:
\begin{itemize}[leftmargin=*,itemsep=1pt,topsep=2pt]
\item It splits the accuracy that FL loses on real hospitals into a part caused by imaging devices and the rest, on two datasets, and traces the device part to the batch-norm statistics that the server averages.
\item It shows that a layer-norm backbone avoids this part and beats the benchmark baseline, and that a short training budget can hide this advantage.
\item It shows that, under DP, features learned from public images of another hospital make a private federated classifier usable, and measures what it still costs.
\end{itemize}

\section{Related work}
FL loses accuracy when sites hold different data, and batch normalization is known to suffer most. Hsieh et al. [@hsieh] showed that averaging batch-norm layers hurts when labels are skewed across sites, and FedBN [@fedbn] keeps these layers at each site instead. These studies vary the label mix or split public datasets; they do not ask which real difference between hospitals causes the loss. For privacy, DP-SGD [@abadi] trains with clipped and noisy gradients, Tramèr and Boneh [@tramer] showed that under DP a linear classifier on good fixed features is a strong choice, and RMIA [@rmia] is a membership attack that stays strong with few reference models.

Table~\ref{tab:cmp} compares this work with the two studies closest to it, one for each of our datasets. FLamby is our baseline. We reproduce its EfficientNet-B0 with FedAvg on Fed-ISIC2019 and obtain 0.624, against 0.539 with a standard deviation of 0.097 over five seeds in the released results, so our run lies within their spread. FLamby trains for a short, fixed budget, and Section~\ref{sec:best} shows that this budget hides the advantage of the layer-norm model. Chen et al. [@chen] also build clients from real dermoscopy sources, but on three classes and with macro F1, so their numbers cannot be compared with ours directly. Neither study divides the accuracy loss by cause or measures privacy, and these are the two parts this work adds.

@TABLE1@

\section{Method}
Figure~\ref{fig:pipe} shows the pipeline that every experiment runs; experiments differ only in what fills each box. The training images are split into $K$ sites. At each site, images are preprocessed, augmented and passed through a backbone, a stack of blocks each followed by a normalization layer, and then through a linear classifier. The local update trains this network on the site's own images for one epoch. Each site then sends its model to the server, which averages the $K$ models, weighted by site size, into a global model that goes back to every site for the next round. After 30 rounds the final model is scored for accuracy and attacked for membership. Centralized training, the reference for every accuracy gap, uses the same boxes in one place that holds all images. The study changes three boxes: the site split, the normalization layer and, for privacy, the local update, whose private form is shown in Figure~\ref{fig:design}.

@FIG1@

@FIG3@

\subsection{Sites and the device-mixing control}
HAM10000 has 10,015 dermoscopy images of 7,470 lesions in seven classes, from four sources: three collections of the Department of Dermatology in Vienna, which we call molemax, modern and dias, and one practice in Queensland, Rosendahl. We use these sources as sites because each is a real clinic or imaging setup. They differ nine-fold in size, from 365 to 3,388 training images, and in disease mix: 94\% of the molemax images are benign nevi, and two sites have no image of actinic keratosis. Training and validation images are split by lesion, so that no lesion appears in both, which gives 8,583 training and 1,432 validation images. The official ISIC 2018 test set of 1,512 images is scored once per final model. Since 67\% of the images are nevi, we report balanced accuracy, the mean recall over the seven classes.

The images differ more than the labels. Figure~\ref{fig:nevi} shows benign nevi from each site: colour, skin tone and dermoscope borders change with the source. From six colour statistics of nevi alone, a logistic regression finds the source with accuracy 0.80, where chance is 0.25.

@FIG2@

The device-mixing control asks whether these device differences cost accuracy in FL. It builds a second set of four sites from the same training images. Each mixed site has nearly the same number of images per class as the real site it replaces, but draws its lesions at random from all four sources. Site size, disease mix and the images stay the same and only the devices behind each site change, so any difference in accuracy between real and mixed sites comes from devices. We call the accuracy that FedAvg on the real sites loses against centralized training the FL gap, and the part of it that mixing recovers the device share.

Fed-ISIC2019 has 23,247 images in eight classes from six centres, four of which are the HAM10000 sources. There we mix devices in the same way and use the FLamby metric, the mean of the balanced accuracy of each centre.

\subsection{Backbone and normalization}
The backbone is where we look for the device effect, so we vary its normalization layer. A batch-norm layer, or BN, normalizes each channel with a mean and variance estimated from the training images and stores these statistics in the model. A layer-norm layer, or LN, normalizes each image with its own features and stores nothing. In FL the server averages the stored BN statistics of all sites like any other weight. If the devices of each site shift its image statistics, BN is the layer that records the shift, and the averaged statistics fit no site.

We compare two BN models, EfficientNet-B0 [@tan] and ResNet-18, two LN models, ConvNeXt-atto [@liu] and DeiT-tiny, and a ResNet-50 that exists with BN and with group normalization under the same architecture and pretraining, so that its two versions differ only in this layer. EfficientNet-B0 is the baseline because it is the FLamby model. ConvNeXt-atto is our main LN model because it is a convolutional network of similar size, 3.4 million weights against 4.0 million, so it differs from the baseline in its layers rather than in scale.

All models start from ImageNet weights and train at 224 pixels with AdamW, learning rate 0.0003, batch size 64 and class-weighted cross-entropy; the ResNet-50 pair runs at 160 pixels to fit a 4 GB GPU. Federated runs use 30 rounds of one local epoch and keep the round with the best validation accuracy. Each setting runs with three seeds, and five for the baseline and the best model.

\subsection{Privacy}
For privacy we first measure, then protect. The attacker is the server, or anyone who later gets the model, and tries to tell whether a given image was used in training. We use RMIA [@rmia] with eight reference models per backbone and report its true-positive rate at a 1\% false-positive rate: the share of training images the attacker finds while wrongly flagging only one in a hundred other images.

To protect, each site trains with DP-SGD [@abadi]: the gradient of each image is clipped and Gaussian noise is added at every step, and each site spends its own privacy budget $\varepsilon$ of 8, 3 or 1 over all rounds. We apply DP to the last one or two blocks and to the classifier alone. Noise spread over millions of weights drowns the signal, so the private design trains only the linear classifier on top of a frozen backbone, about nine thousand weights, and the frozen features then decide how accurate it can be [@tramer]. With ImageNet features this is our private baseline. In the proposed design we instead pretrain the same EfficientNet-B0 on the BCN images of ISIC 2019, public dermoscopy images from a hospital outside the federation, after removing any image that duplicates a HAM10000 image. The design differs from the private baseline only in these frozen weights.


\section{Results}
Numbers are validation balanced accuracy on HAM10000, averaged over seeds, unless stated otherwise.

\subsection{Devices cause a large part of the FL gap}
Removing device differences recovers 35\% to 49\% of the FL gap for the three BN models and none of it for the two LN models (Table~\ref{tab:dev}). For BN models, FedAvg loses 0.10 to 0.14 against centralized training, and mixing wins back about 0.05 of it. For LN models the gap is smaller, 0.076 and 0.036, and mixing does not raise accuracy.

@TABLE2@

The loss falls on the class that matters most. With the EfficientNet-B0 baseline, FL lowers melanoma recall on the test set from 0.645 to 0.380, as the model labels more images as benign nevi, the largest class. Mixing devices raises it back to 0.499. ConvNeXt-atto keeps most of it, 0.692 against 0.756 when trained centrally.

\subsection{The device effect enters through the BN statistics}
Devices could hurt FL in two ways: by turning the updates of different sites in different directions, or by shifting the BN statistics that the server averages. The first is ruled out: mixing raised the accuracy of EfficientNet-B0 by 0.047 while the cosine similarity between site updates changed by at most 0.05. Four results point to the second. Mixing helps all three BN models and neither LN model. In the ResNet-50 pair, which differs only in normalization, mixing gains 0.051 with BN in every seed but only 0.014 with group normalization, within noise. FedBN, which keeps the BN statistics at each site, gains 0.080 over FedAvg on the real sites but only 0.013 on the mixed sites. Finally, colour correction of each image raises FedAvg by 0.024 for EfficientNet-B0 but only 0.006 for ConvNeXt-atto.

FedBN itself is not a practical fix, because it needs to know which site a test image comes from; without that it scores 0.639, below FedAvg. About 60\% of the gap of BN models remains after mixing. It comes with the differences in disease mix and site size, and no method we tried reduced it.

\subsection{The best federated model and the baseline}\label{sec:best}
Avoiding BN gives the best federated model. ConvNeXt-atto with FedAvg reaches 0.741 with a standard deviation of 0.013 over five seeds, against 0.661 with 0.010 for the EfficientNet-B0 baseline, and 0.662 against 0.597 on the test set (Table~\ref{tab:designs}). Trained centrally, the two differ by only 0.035, so most of the difference appears only in FL. The baseline levels off within ten rounds, while ConvNeXt-atto keeps improving until the last rounds. It is also smaller and trains faster, about 9 minutes per run against 14 on a 4 GB laptop GPU.

@TABLE3@

The device effect repeats on Fed-ISIC2019. Under the FLamby protocol, mixing raises FedAvg of the baseline by 0.030, a device share of 28\%. With three times the training steps the share is 42\%, close to the 40\% on HAM10000, while for ConvNeXt-atto mixing adds only 0.008. The LN advantage needs this longer budget. At the FLamby budget the two models are close, 0.643 and 0.628; with three times the steps ConvNeXt-atto leads by 0.043 in the FLamby metric. On all test images pooled the two tie, because ConvNeXt-atto gains mostly at the small centres.

\subsection{Adding privacy}
Federated training lowers leakage but does not remove it. At a 1\% false-positive rate, RMIA finds 4.3\% of the training images of the federated baseline, against 9.6\% when it is trained centrally, and 5.4\% against 7.1\% for ConvNeXt-atto. A formal guarantee therefore still needs DP.

DP works only on the classifier. With DP on the last one or two blocks, accuracy falls to between 0.15 and 0.24 at every budget, close to the 0.14 of random guessing. On the classifier with ImageNet features, accuracy falls from 0.617 without DP to 0.430 at $\varepsilon = 3$, and most of this loss comes from the noise rather than from clipping. The best federated backbone does not help here: frozen ConvNeXt-atto features give 0.353 at $\varepsilon = 3$, below EfficientNet-B0, so the private designs keep EfficientNet-B0.

Public dermoscopy features raise private accuracy by 0.12 to 0.13 at every budget (Figure~\ref{fig:priv}), from 0.430 to 0.554 at $\varepsilon = 3$. Without DP, a classifier on these features already reaches 0.710, above the full FedAvg baseline. The gain does not come from shared images: none of the 7,463 candidate pairs flagged between BCN and HAM10000 was a confirmed duplicate. The trade-off narrows but remains, as the private design stays 0.19 below the best federated model.

@FIG4@

\section{Discussion and limitations}
For hospitals that build a federated skin-lesion model, the normalization of the backbone mattered more than the FL algorithm. With BN, part of the accuracy lost to FL comes from device differences and falls hardest on melanoma, and an LN backbone avoids that part without needing to know where each image comes from. Benchmarks should also report their training budget, since the FLamby budget hid this advantage. For privacy, keeping images on site is not enough. If a formal guarantee is needed, DP at this scale works only on the classifier, and good public features decide how much accuracy is left. A hospital must still choose between the more accurate model and the one with a guarantee, 0.19 apart.

These conclusions have limits. The federation is small, four sites with 8,583 training images, so most differences between single sites are not significant. The four comparisons point to the BN statistics but do not prove that they are the only path, and apart from the ResNet-50 pair, BN and LN models also differ in architecture. For the LN models, mixing did not simply help: it moved the errors of ConvNeXt-atto between classes, lowering its melanoma recall to 0.554 at the same balanced accuracy, and it lowered DeiT-tiny for reasons we could not find. The attack uses reference models trained centrally, so the measured leakage is a lower bound, and the public images come from the same ISIC collection as HAM10000.

\section{Conclusion}
We measured where federated learning loses accuracy on real skin-lesion sites and what a formal privacy guarantee costs. For batch-norm models, imaging devices cause about 40\% of the FL gap, through the statistics that the server averages. A layer-norm backbone avoids this and gives the best federated model, 0.741 against 0.661 for the baseline, and the device effect repeats on Fed-ISIC2019. Under DP, public dermoscopy features raise accuracy at $\varepsilon = 3$ from 0.430 to 0.554, still 0.19 below the best model. Code, data splits and run logs are available from the author on request.
"""

ABSTRACT = r"""Federated learning lets hospitals train one skin-lesion classifier without sharing images, but on real hospitals it is less accurate than training on all images in one place, and the trained model can still reveal which images it was trained on. We study where this accuracy goes and what a formal privacy guarantee costs, using the four image sources of HAM10000 as hospitals and the six centres of Fed-ISIC2019 as a second test. Mixing imaging devices across sites, while keeping each site's images and disease mix, recovers 35\% to 49\% of the accuracy gap of batch-norm models, and four controlled comparisons point to the batch-norm statistics that the server averages as the reason. A layer-norm backbone avoids this loss: ConvNeXt-atto reaches 0.741 balanced accuracy, against 0.661 for the EfficientNet-B0 baseline of the FLamby benchmark, and the device effect repeats on Fed-ISIC2019. For privacy, federated models still leak under a strong membership attack, and differential privacy works only on the last layer, where features pretrained on public dermoscopy images from another hospital raise accuracy at $\varepsilon = 3$ from 0.43 to 0.55."""

BANG = {
    "TABLE1": r"""\begin{table*}[t]
\caption{This work and the two closest studies.}\label{tab:cmp}
\centering\small
\begin{tabular}{lllll}
\toprule
Study & Data & Sites & Accuracy reported & Privacy measured \\
\midrule
FLamby [@flamby] & Fed-ISIC2019, 8 classes & 6 centres & mean over centres & no \\
Chen et al. [@chen] & HAM10000 and MSK, 3 classes & 4 sources & macro F1 and AUC & no \\
This work & HAM10000 and Fed-ISIC2019 & 4 sources, 6 centres & gap split by cause & attack and DP \\
\bottomrule
\end{tabular}
\end{table*}""",
    "TABLE2": r"""\begin{table}[t]
\caption{Mixing devices recovers part of the FL gap for batch-norm models only. Real and mixed: FedAvg on the real and on the device-mixed sites. Share: device share.}\label{tab:dev}
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llccccc}
\toprule
Backbone & Norm & Central & Real & Mixed & Gap & Share \\
\midrule
EfficientNet-B0 & BN & 0.778 & 0.659 & 0.707 & 0.118 & 40\% \\
ResNet-18 & BN & 0.797 & 0.657 & 0.706 & 0.140 & 35\% \\
ResNet-50 & BN & 0.800 & 0.698 & 0.748 & 0.102 & 49\% \\
ResNet-50 & GN & 0.791 & 0.700 & 0.713 & 0.092 & 15\% \\
ConvNeXt-atto & LN & 0.813 & 0.737 & 0.733 & 0.076 & none \\
DeiT-tiny & LN & 0.779 & 0.742 & 0.698 & 0.036 & none \\
\bottomrule
\end{tabular}
\end{table}""",
    "TABLE3": r"""\begin{table}[t]
\caption{Final designs on HAM10000. The DP guarantee is at $\varepsilon = 3$.}\label{tab:designs}
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccc}
\toprule
Design & Weights trained & Val. & Test & DP \\
\midrule
Baseline, EfficientNet-B0 & 4.0 million & 0.661 & 0.597 & no \\
Best federated, ConvNeXt-atto & 3.4 million & 0.741 & 0.662 & no \\
Private, ImageNet features & 8,967 & 0.430 & 0.386 & yes \\
Private, public features & 8,967 & 0.554 & 0.508 & yes \\
\bottomrule
\end{tabular}
\end{table}""",
}
HINH = {
    "FIG1": (r"figure*", "fig1_pipeline.pdf", r"\textwidth",
             "Pipeline shared by all experiments. The study changes the site split, the normalization layer and the local update."),
    "FIG2": (r"figure", "fig2_nevi.pdf", r"\columnwidth",
             "Benign nevi from the four HAM10000 sources. The diagnosis is the same in every row; colour, skin tone and borders change with the source."),
    "FIG3": (r"figure*", "fig3_design.pdf", r"\textwidth",
             "Proposed private design. A backbone pretrained on public images is frozen; each hospital trains only the classifier with DP-SGD, and the server averages the classifiers."),
    "FIG4": (r"figure", "fig4_privacy.pdf", r"\columnwidth",
             "Accuracy of the private design at each privacy budget, with ImageNet and with public dermoscopy features. The dashed line is the best federated model without DP."),
}
NHAN = {"FIG1": "fig:pipe", "FIG2": "fig:nevi", "FIG3": "fig:design", "FIG4": "fig:priv"}
for k, (moi, f, w, cap) in HINH.items():
    TEX = TEX.replace(f"@{k}@", f"\\begin{{{moi}}}[t]\n\\centering\n\\includegraphics[width={w}]{{figures/{f}}}\n"
                                f"\\caption{{{cap}}}\\label{{{NHAN[k]}}}\n\\end{{{moi}}}")
for k, v in BANG.items():
    TEX = TEX.replace(f"@{k}@", v)

thu_tu = []
for k in re.findall(r"\[@(\w+)\]", TEX):
    if k not in thu_tu:
        thu_tu.append(k)
so = {k: i + 1 for i, k in enumerate(thu_tu)}
TEX = re.sub(r"\[@(\w+)\]", lambda m: f"[{so[m.group(1)]}]", TEX)
assert set(thu_tu) == set(KHOA), set(KHOA) ^ set(thu_tu)
TL = "\n".join(rf"\bibitem{{r{so[k]}}} {tai_lieu(KHOA[k])}" for k in thu_tu)

DAU = r"""\documentclass[10pt,twocolumn,a4paper]{article}
\usepackage[top=2cm,bottom=2.2cm,left=1.8cm,right=1.8cm,columnsep=0.6cm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[font=small,labelsep=colon,skip=4pt]{caption}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{10pt}{4pt}
\titlespacing*{\subsection}{0pt}{8pt}{3pt}
\usepackage{url}
\usepackage{xurl}
\urlstyle{same}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{1em}
\setcounter{dbltopnumber}{3}
\renewcommand{\dbltopfraction}{0.85}
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\setlength{\textfloatsep}{10pt}
\setlength{\dbltextfloatsep}{10pt}
\renewcommand{\refname}{References}
\begin{document}
\twocolumn[
\begin{center}
{\Large\bfseries Privacy-preserving Medical Image Analysis using Federated Learning:\\[3pt] Device Effects and the Cost of Privacy on Real Skin-Lesion Sites\par}
\vspace{10pt}
{\large Đoàn Đức Nguyên (0009-0005-7460-8494)\par}
\vspace{2pt}
Department of Information Technology, FPT University\par
\vspace{12pt}
\begin{minipage}{0.86\textwidth}
\noindent\textbf{Abstract.} """ + ABSTRACT + r"""
\end{minipage}
\vspace{16pt}
\end{center}
]
"""
CUOI = r"""
{\small
\begin{thebibliography}{99}
\setlength{\itemsep}{1pt}
""" + TL + r"""
\end{thebibliography}}
\end{document}
"""
(D / "paper.tex").write_text(DAU + TEX + CUOI)
print("ok", len(thu_tu), "tài liệu")
