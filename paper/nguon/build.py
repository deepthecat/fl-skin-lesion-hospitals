"""Chuyển nội dung doc (doc.txt: dòng H/P/C) sang LaTeX IEEEtran, biên dịch bằng xelatex."""
import re
import shutil
from pathlib import Path

D = Path(__file__).parent
FIG = D / "figures"
FIG.mkdir(exist_ok=True)
NGUON_HINH = {
    1: ("/home/claude/hinh_moi/v2/hinh_paper/hinh1_khung.pdf", "fig1_pipeline.pdf", True),
    2: ("/home/claude/hinh_moi/v2/hinh_paper/hinh2_nevi.pdf", "fig2_nevi.pdf", False),
    3: ("/home/claude/hinh_moi/v2/hinh_paper/hinh3_thiet_ke.pdf", "fig3_design.pdf", True),
    4: ("/home/claude/hinh_moi/that/hinh_paper/hinh4_chuan_hoa.pdf", "fig4_device_mixing.pdf", False),
    5: ("/home/claude/hinh_moi/that/hinh_paper/hinh6_duong_hoc.pdf", "fig5_curves.pdf", False),
    6: ("/home/claude/hinh_moi/that/hinh_paper/hinh5_rieng_tu.pdf", "fig6_privacy.pdf", False),
}
for n, (src, dst, _) in NGUON_HINH.items():
    shutil.copy(src, FIG / dst)

SO_COT = {"I": 7, "II": 5, "III": 7, "IV": 5, "V": 6, "VI": 5}


def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"$\sim$"), ("^", r"\^{}")):
        s = s.replace(a, b)
    s = s.replace("−", "$-$").replace("+/-", r"$\pm$")
    s = re.sub(r"(?<![A-Za-z])epsilon(?![A-Za-z])", r"$\\varepsilon$", s)
    s = re.sub(r"\bmu = ", r"$\\mu$ = ", s)
    s = re.sub(r"\bbeta = ", r"$\\beta$ = ", s)
    s = s.replace("delta = 1e-5", r"$\delta = 10^{-5}$")
    s = re.sub(r"(\d)e-(\d)", r"\1$\\times 10^{-\2}$", s)
    s = re.sub(r"(\d) x (\d)", r"\1$\\times$\2", s)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    s = re.sub(r"(?<![\w\-.{])-(\d)", r"$-$\1", s)
    s = s.replace("]-[", "]--[").replace("Jensen-Shannon", "Jensen--Shannon").replace("privacy-accuracy", "privacy--accuracy")
    s = re.sub(r"\$\\(varepsilon|mu|beta)\$ = (\d+(?:\.\d+)?)", r"$\\\1 = \2$", s)
    for a, b in (("C = 7 labels", "$C = 7$ labels"), ("K = 4 sites, and site k holds n\\_k of the n images", "$K = 4$ sites, and site $k$ holds $n_k$ of the $n$ images"),
                 ("with p = 6,", "with $p = 6$,"), ("C = 8 and K = 6", "$C = 8$ and $K = 6$"), ("Renyi-DP", "R\\'enyi-DP")):
        s = s.replace(a, b)
    for n, c in enumerate("①②③④", 1):
        s = s.replace(c, rf"\mk{{{n}}}")
    s = re.sub(r"\b(vidir\\_molemax|vidir\\_modern|vienna\\_dias|rosendahl)\b", r"\\texttt{\1}", s)
    return s


def bang(so, tieu_de, o):
    k = SO_COT[so]
    hang = [o[i:i + k] for i in range(0, len(o), k)]
    rong = so in ("I", "III", "V", "VI")
    moi = "table*" if rong else "table"
    cot = {"I": "lllcccc", "II": "lrrrr", "IV": "llccc", "V": "lccccc"}.get(so, "l" + "c" * (k - 1))
    if so == "IV":
        cot = "@{}llccc@{}"
    r = [rf"\begin{{{moi}}}[!t]", r"\centering", rf"\caption{{{esc(tieu_de)}}}", rf"\label{{tab:{so}}}",
         r"\footnotesize" if not rong else r"\small", rf"\begin{{tabular}}{{{cot}}}", r"\hline"]
    r.append(" & ".join(esc(x) for x in hang[0]) + r" \\ \hline")
    for n_, h in enumerate(hang[1:]):
        if so == "I" and h[0] == "This work":
            r.append(r"\hline")
        r.append(" & ".join(esc(x) for x in h) + r" \\")
    r += [r"\hline", r"\end{tabular}", rf"\end{{{moi}}}", ""]
    return "\n".join(r)


def hinh(n, chu_thich):
    _, dst, rong = NGUON_HINH[n]
    moi = "figure*" if rong else "figure"
    w = r"\textwidth" if rong else r"\columnwidth"
    return "\n".join([rf"\begin{{{moi}}}[!t]", r"\centering", rf"\includegraphics[width={w}]{{figures/{dst}}}",
                      rf"\caption{{{esc(chu_thich)}}}", rf"\label{{fig:{n}}}", rf"\end{{{moi}}}", ""])


EQ1 = r"""\begin{equation}
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\, w_{t+1}^{k}
\end{equation}
"""
EQ2 = r"""\begin{equation}
\mathrm{BAcc} = \frac{1}{C} \sum_{c=1}^{C} R_c
\end{equation}
"""

SUA = {
    "including the Fed-ISIC2019 benchmark of FLamby [9] and [10],": "including the Fed-ISIC2019 benchmark of FLamby [9] and the study of Chen et al. [10],",
    "Code, splits and run logs: [link].": "Code, data splits and run logs are available from the author on request.",
    " [kiểm trang]": "",
}
nd = (D / "doc_v9.txt").read_text()
for a, b in SUA.items():
    nd = nd.replace(a, b)
dong = [l.split("\t", 1) for l in nd.split("\n")]

DONG_GOP = r"""This work makes three contributions:
\begin{itemize}
\item We split the FL accuracy gap on real hospitals into a part caused by imaging devices and the rest, on two datasets, and trace the device part to the batch-normalization statistics that the server averages.
\item We show that a layer-norm backbone avoids this part and gives a better federated model than the benchmark baseline, and that a short training budget can hide this advantage.
\item We measure membership leakage with a strong attack and show that, under DP, features learned from public images of another hospital make a private federated classifier usable, at a measured cost against the best non-private design.
\end{itemize}
"""
GIA_THUYET = r"""Five hypotheses are tested:
\begin{itemize}
\item[] \textit{M1:} devices hurt FL by changing the direction of local updates;
\item[] \textit{M2:} devices hurt FL through the BN statistics that FedAvg averages;
\item[] \textit{A:} under DP at the same $\varepsilon$, an LN model beats the best BN model;
\item[] \textit{H1:} under DP, the smallest site and the rare classes lose the most accuracy;
\item[] \textit{H2:} without DP, the smallest site and the rare classes leak the most.
\end{itemize}
"""
out, i, tom_tat, tu_khoa, tai_lieu = [], 0, "", "", []
che_do = None
while i < len(dong):
    loai, txt = dong[i][0], (dong[i][1] if len(dong[i]) > 1 else "")
    if loai == "H":
        if txt == "Abstract":
            che_do = "abstract"
        elif txt == "References":
            che_do = "ref"
        elif txt == "Reproducibility":
            che_do = None
            out.append(r"\section*{Reproducibility}")
        elif re.match(r"^[IVX]+\. ", txt):
            che_do = None
            out.append(rf"\section{{{esc(txt.split('. ', 1)[1])}}}")
        elif re.match(r"^[A-H]\. ", txt):
            out.append(rf"\subsection{{{esc(txt.split('. ', 1)[1])}}}")
        i += 1
        continue
    if che_do == "abstract":
        if txt.startswith("Index Terms:"):
            tu_khoa = txt.split(":", 1)[1].strip().rstrip(".")
        elif txt:
            tom_tat = txt
        i += 1
        continue
    if che_do == "ref":
        m = re.match(r"^\[(\d+)\]\s*(.*)$", txt)
        if m:
            tai_lieu.append((m.group(1), m.group(2)))
        i += 1
        continue
    if loai == "P" and txt == "":
        i += 1
        continue
    m = re.match(r"^TABLE ([IVX]+)\.\s*(.*)$", txt)
    if m:
        so, td = m.group(1), m.group(2)
        o = []
        i += 1
        while i < len(dong) and dong[i][0] == "C":
            o.append(dong[i][1])
            i += 1
        out.append(bang(so, td, o))
        continue
    m = re.match(r"^Fig\. (\d)\. (.*)$", txt)
    if m:
        out.append(hinh(int(m.group(1)), m.group(2)))
        i += 1
        continue
    if txt.startswith("This work makes three contributions."):
        out.append(DONG_GOP)
        i += 1
        continue
    if " Five hypotheses are tested: " in txt:
        dau = txt.split(" Five hypotheses are tested: ")[0]
        out.append(esc(dau) + "\n" + GIA_THUYET)
        i += 1
        continue
    if txt.startswith("RUNIN:"):
        dau, than = txt[6:].split("|", 1)
        out.append(r"\textit{" + esc(dau) + "} " + esc(than) + "\n")
        i += 1
        continue
    if txt.startswith("FOOTNOTE:"):
        out[-1] = out[-1].rstrip("\n") + r"\footnote{" + esc(txt[9:].strip()) + "}\n"
        i += 1
        continue
    out.append(esc(txt.strip()) + "\n")
    if txt.endswith("in proportion to site size:"):
        out.append(EQ1)
    if txt.endswith("ISIC 2018 metric [18]:"):
        out.append(EQ2)
    i += 1

# Liên kết chính thức cho tài liệu không có DOI (đã kiểm từng cái ngày 10.11.2026)
LINK = {
    "2": "https://proceedings.mlr.press/v54/mcmahan17a.html",
    "3": "https://arxiv.org/abs/1909.06335",
    "4": "https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html",
    "9": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract.html",
    "12": "https://proceedings.mlr.press/v235/zarifzadeh24a.html",
    "13": "https://openreview.net/pdf/37e144ecec7a76ce5093d4ec328649e8ef208a5b.pdf",
    "14": "https://proceedings.mlr.press/v119/hsieh20a.html",
    "16": "https://openaccess.thecvf.com/content/CVPR2022/html/Qu_Rethinking_Architecture_Design_for_Tackling_Data_Heterogeneity_in_Federated_Learning_CVPR_2022_paper.html",
    "17": "https://openreview.net/forum?id=GtdYFLsblb",
    "18": "https://arxiv.org/abs/1902.03368",
    "20": "https://openaccess.thecvf.com/content_CVPRW_2019/html/ISIC/Bissoto_DeConstructing_Bias_on_Skin_Lesion_Datasets_CVPRW_2019_paper.html",
    "23": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ea46207ec9bda974b140fe11d8dd727-Abstract-Conference.html",
    "24": "https://proceedings.mlr.press/v252/tavakoli24a.html",
    "28": "https://openreview.net/forum?id=YTWGvpFOQD-",
    "30": "https://proceedings.neurips.cc/paper/2019/hash/fc0de4e0396fff257ea362983c2dda5a-Abstract.html",
    "31": "https://proceedings.mlr.press/v97/tan19a.html",
    "32": "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
    "33": "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
    "34": "https://proceedings.mlr.press/v139/touvron21a.html",
    "35": "https://openreview.net/forum?id=Bkg6RiCqY7",
}
SUA_TL = {
    # Bissoto: bỏ số trang và DOI chưa kiểm được, dùng trang chính thức của CVF
    "20": (", pp. 2766-2774, doi: 10.1109/CVPRW.2019.00335.", "."),
    # Shades of Gray: thêm DOI đã kiểm trên library.imaging.org
    "22": ("2004, pp. 37-41.", "2004, pp. 37-41, doi: 10.2352/CIC.2004.12.1.art00008."),
}
import json as _json
CU_CUA = _json.loads((D / "ref_map.json").read_text())   # số mới -> số cũ
tai_lieu = [(n, t.replace(*SUA_TL[CU_CUA[n]]) if CU_CUA[n] in SUA_TL and SUA_TL[CU_CUA[n]][0] in t else t) for n, t in tai_lieu]
assert all(SUA_TL[CU_CUA[n]][1] in t for n, t in tai_lieu if CU_CUA[n] in SUA_TL)


def esc_ref(s, n):
    s = re.sub(r"pp\. (\d+)-(\d+)", r"pp. \1@@\2", s)
    if '," ' in s:
        dau, duoi = s.split('," ', 1)
        mo = ""
        if duoi.startswith("in "):
            mo, duoi = "in ", duoi[3:]
        m = re.match(r"^(.*?)(, (?:vol\.|PMLR|no\.|pp\.|art\.|Datasets|\d{4}).*)$", duoi)
        if m and not duoi.startswith("arXiv"):
            s = esc(dau + '," ') + esc(mo) + r"\textit{" + esc(m.group(1)) + "}" + esc(m.group(2))
        else:
            s = esc(s)
    else:
        s = esc(s)
    s = s.replace("@@", "--")
    # DOI -> liên kết tuyệt đối https://doi.org/... (bản cũ để \url{10.x/...} nên trình đọc PDF hiểu là đường dẫn file tương đối)
    s = re.sub(r"doi: (\S+?)\.?$", lambda m: r"doi: \href{https://doi.org/" + m.group(1).replace(r"\_", "_") + "}{"
               + m.group(1) + "}.", s)
    if CU_CUA[n] in LINK:
        s = s.rstrip(".") + r". [Online]. Available: \url{" + LINK[CU_CUA[n]] + "}"
    return s


tex = r"""\documentclass[conference]{IEEEtran}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{url}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\usepackage{balance}
\usepackage{tikz}
\DeclareRobustCommand{\mk}[1]{\tikz[baseline=(c.base)]\node[draw,circle,inner sep=0.5pt,minimum size=7pt,font=\scriptsize\bfseries](c){#1};}
\raggedbottom
\begin{document}
\title{Privacy-preserving Medical Image Analysis using Federated Learning: Device Effects and the Cost of Privacy on Real Skin-Lesion Sites}
\author{\IEEEauthorblockN{Đoàn Đức Nguyên (0009-0005-7460-8494)}
\IEEEauthorblockA{Department of Information Technology, FPT University}}
\maketitle
\begin{abstract}
""" + esc(tom_tat) + r"""
\end{abstract}
\begin{IEEEkeywords}
""" + esc(tu_khoa) + r"""
\end{IEEEkeywords}

""" + "\n".join(out) + r"""
\makeatletter\def\IEEEbibitemsep{0pt plus .5pt}\makeatother
\begin{thebibliography}{99}
""" + "\n".join(rf"\bibitem{{r{n}}} {esc_ref(t, n)}" for n, t in tai_lieu) + r"""
\end{thebibliography}
\end{document}
"""
(D / "paper.tex").write_text(tex)
print("ok", len(tai_lieu), "refs")
