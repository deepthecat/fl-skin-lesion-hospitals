# Why Federated Learning Falls Short Across Hospitals

Code, run results and paper for a study of federated learning on skin-lesion images.
Đoàn Đức Nguyên ([ORCID 0009-0005-7460-8494](https://orcid.org/0009-0005-7460-8494)), Department of Information Technology, FPT University.

Paper: [paper/Paper_FL_HAM10000.pdf](paper/Paper_FL_HAM10000.pdf)

## What the study asks
Federated learning (FL) lets hospitals train one model without sharing patient images. On real hospital data,
it is usually less accurate than training on all images in one place, and a trained model can still reveal which
images were used to train it. The study asks two questions:

1. Where is the accuracy lost when hospitals train together with FedAvg?
2. How much accuracy does a formal privacy guarantee (differential privacy) cost?

It treats the four image sources of HAM10000 as four hospitals and uses the six centres of Fed-ISIC2019 as a second test.

## Main results
- When the sites are rebuilt so that each mixes images from all sources, with the same number of images per disease,
  models with batch normalization recover 35% to 49% of the accuracy they lose in federated training.
  Four further comparisons support the hypothesis that this loss enters through the batch-normalization
  statistics that the server averages.
- A layer-normalization model loses less: ConvNeXt-atto reaches 0.741 balanced accuracy in federated training,
  against 0.661 for EfficientNet-B0, the model of the FLamby benchmark.
- The RMIA membership attack still finds training images in federated models. With differential privacy,
  only training the last layer kept useful accuracy, and features learned from public images of another hospital
  raised accuracy at privacy level ε = 3 from 0.43 to 0.55.

Image data and model weights are not included (size and licence; HAM10000 and ISIC 2019 images are CC BY-NC).

## Layout
| Folder | Content |
|---|---|
| `paper/` | PDF, LaTeX source, the four figures, and in `scripts/` the code that draws the figures and checks every number in the paper |
| `notebooks/` | 01 to 06: data audit, lesion-level split, image sources, preprocessing, image EDA, site split |
| `src/` | All training, attack and analysis scripts (see the pipeline below) |
| `results/` | One JSON file per run (same name as its wandb run) and the summary CSV files |
| `run_history/` | The shell scripts that launched each night of runs, with the exact commands, plus the git log |
| `requirements.txt` | Python packages of the environment used |

Names of scripts and variables are in Vietnamese. Short glossary: `tap_trung` centralized, `lien_ket` federated,
`nguon` real sites, `tron_nguon` mixed sites, `txlp1`/`txlp2` preprocessing P1/P2, `kt224` image size 224,
`s0`..`s4` seed, `dptt` DP on the linear classifier, `dpsau` DP on deeper blocks, `bcn` features pretrained on BCN images.

## Environment
One laptop GPU with 4 GB of memory, Python 3 with PyTorch and timm, scikit-learn, Opacus (only its RDP accountant),
and optionally wandb. Data are expected under `~/ham_fl/data` (or set `HAM_GOC`); results are written to `~/ham_fl/ketqua`.

Data sources: HAM10000 (Tschandl et al., Sci. Data 2018) with the official ISIC 2018 test set; ISIC 2019 with the
Fed-ISIC2019 split of FLamby (Ogier du Terrail et al., NeurIPS 2022).

## Pipeline, in order
1. `notebooks/01`..`06`: check the data, split training and validation by lesion (`chia_tap.csv`), attach the source of
   each image (`nguon.csv`), compare preprocessing, and build the site columns.
2. `src/tao_kich_ban_tron.py`: build the mixed sites (column `may_tron_nguon`).
3. Centralized reference: `python huan_luyen_tap_trung.py --backbone efficientnet_b0 --kt 224 --epoch 15 --tien-xu-ly p1 --seed 0`
4. Federated training on real or mixed sites:
   `python huan_luyen_lien_ket.py --thuat-toan fedavg --kich-ban nguon --backbone convnext_atto --tien-xu-ly p1 --seed 0`
   (use `--kich-ban tron_nguon` for mixed sites, `--thuat-toan fedbn` or `fedprox` for the other algorithms).
5. Fed-ISIC2019: `chuan_bi_isic.py` (images as in FLamby), `tao_tron_isic.py` (mixed centres), `fed_isic.py`
   (FLamby protocol and the longer budgets; exact flags in `run_history/`).
6. Membership attacks: `huan_luyen_tham_chieu.py --cap 0..3 --nua 0/1` (eight reference models per backbone),
   then `tan_cong_thanh_vien.py` (loss attack) and `rmia.py` (RMIA).
7. DP: `dp_sau.py` (DP on the last one or two blocks) and `dp_tuyen_tinh.py` (DP on the linear classifier,
   e.g. `--backbone efficientnet_b0 --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr 4`).
   Proposed version: pretrain on BCN with `fed_isic.py --che-do cuc_bo --tram 0 --anh goc --backbone efficientnet_b0 --cong-thuc ham --loai-tru ketqua/bcn_loai_tru.csv --luu-mo-hinh`,
   then `dp_tuyen_tinh.py ... --trong-so-backbone <saved weights> --nhan-dt bcn`.
8. Analysis and paper: `recall_theo_lop.py`, `kiem_do_nguon.py`, `kiem_tra_wandb.py` (checks paper numbers against
   the JSON files and wandb), `paper/scripts/kiem_so_lieu_bang.py` and `paper/scripts/kiem_so_lieu_chu.py` (check every number in the tables and text),
   `paper/scripts/ve_hinh_paper_v4.py` and `paper/scripts/ve_so_do_v4.py` (figures).

## Where each result in the paper comes from
| Paper | Files in `results/` |
|---|---|
| Table 2 | `tap_trung/*_txlp1_*.json`, `lien_ket/fedavg_nguon_*_txlp1_*.json`, `lien_ket/fedavg_tron_nguon_*_txlp1_*.json` |
| Section 4.1, melanoma recall | `recall_theo_lop.csv` |
| Section 4.2 | `lien_ket/fedbn_*`, `lien_ket/*_gh_*` (update directions), `lien_ket/*_txlp2_*` |
| Table 3, Section 4.3 | `lien_ket/fedavg_nguon_{convnext_atto,efficientnet_b0}_txlp1_kt224_s0..s4.json`, `dp_tuyen_tinh_lr4/`, `dp_tuyen_tinh_bcn_lr1/` |
| Section 4.3, Fed-ISIC2019 | `fed_isic/isic_*.json` |
| Section 4.4 and Figure 4 | `rmia.csv`, `dp_sau/`, `dp_tuyen_tinh*/` |

## Experiment logs
Every run is also logged on Weights & Biases, project `ham-fl`, with the same run name as its JSON file.
