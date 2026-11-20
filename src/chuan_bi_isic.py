"""
chuan_bi_isic.py — GĐ1: chuẩn bị ảnh Fed-ISIC2019 ĐÚNG như FLamby (resize_images.py + color_constancy.py).

    python chuan_bi_isic.py

Không cần GPU, chỉ CPU + đĩa (~10–20 phút). Làm 4 việc, mỗi việc tự bỏ qua nếu đã xong:
  1. Giải nén ISIC_2019_Training_Input.zip (25.331 ảnh) vào ~/du_lieu_ngoai/isic2019_raw/.
  2. Đổi cỡ: cạnh ngắn = 224 px, GIỮ tỉ lệ, nội suy BILINEAR — y hệt FLamby.
  3. Hai bản: "cc" = có colour constancy Shades of Gray p=6 (mặc định của FLamby, ~ P2 của mình)
              "goc" = không colour constancy (~ P1) — để kiểm H4 (chuẩn hoá màu có đổi kết luận không) trên bộ thứ hai.
  4. Lấy phép chia train/test theo trung tâm của FLamby (file train_test_split) -> ketqua/fed_isic_chia.csv,
     kiểm số ảnh từng trung tâm khớp README FLamby (9930/2483, 3163/791, 2691/672, 1807/452, 655/164, 351/88).

Colour constancy chép đúng phần tính toán của FLamby (gamma=None nên không cần OpenCV).
"""
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from du_lieu import RA

GOC = Path(os.environ.get("ISIC_DIR", Path.home() / "du_lieu_ngoai" / "isic2019_raw"))
FLAMBY = Path.home() / "FLamby" / "flamby" / "datasets" / "fed_isic2019" / "dataset_creation_scripts"
VAO = GOC / "ISIC_2019_Training_Input"
RA_ANH = {"cc": GOC / "isic2019_224_cc", "goc": GOC / "isic2019_224_goc"}
KY_VONG = {0: (9930, 2483), 1: (3163, 791), 2: (2691, 672), 3: (1807, 452), 4: (655, 164), 5: (351, 88)}
TEN_TT = {0: "BCN", 1: "molemax", 2: "modern", 3: "rosendahl", 4: "MSK", 5: "dias"}   # thứ tự theo README FLamby


def color_constancy(img, power=6):
    """Bản sao phần tính toán color_constancy() của FLamby (gamma=None)."""
    kieu = img.dtype
    img = img.astype("float32")
    rgb = np.power(np.mean(np.power(img, power), (0, 1)), 1 / power)
    rgb = rgb / np.sqrt(np.sum(np.power(rgb, 2.0)))
    rgb = 1 / (rgb * np.sqrt(3))
    return np.multiply(img, rgb).astype(kieu)


def mot_anh(p, sz=224):
    ten = p.name
    if all((RA_ANH[k] / ten).exists() for k in RA_ANH):
        return
    img = Image.open(p)
    r = float(sz) / min(img.size)
    img = img.resize(tuple(int(x * r) for x in img.size), resample=Image.BILINEAR)
    img.save(RA_ANH["goc"] / ten)
    Image.fromarray(color_constancy(np.array(img))).save(RA_ANH["cc"] / ten)


def main():
    z = GOC / "ISIC_2019_Training_Input.zip"
    if not VAO.exists() or len(list(VAO.glob("*.jpg"))) < 25000:
        print(f"1. Giải nén {z.name} ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(GOC)
    anh = sorted(VAO.glob("*.jpg"))
    print(f"1. {len(anh)} ảnh gốc trong {VAO}")

    for d in RA_ANH.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"2–3. Đổi cỡ cạnh ngắn 224 (giữ tỉ lệ) → {RA_ANH['goc'].name} và {RA_ANH['cc'].name} ...")
    Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1))(delayed(mot_anh)(p) for p in tqdm(anh))

    chia = pd.read_csv(FLAMBY / "train_test_split")
    chia["phan"] = chia["fold"].map({"train": "train", "test": "test"})
    chia["trung_tam"] = chia["center"].map(TEN_TT)
    thieu = [i for i in chia.image if not (RA_ANH["cc"] / f"{i}.jpg").exists()]
    print(f"\n4. Phép chia FLamby: {len(chia)} ảnh, thiếu ảnh đã xử lý: {len(thieu)}")
    dem = chia.groupby(["center", "phan"]).size().unstack(fill_value=0)
    loi = False
    for c, (tr, te) in KY_VONG.items():
        a, b = int(dem.loc[c, "train"]), int(dem.loc[c, "test"])
        ok = (a, b) == (tr, te)
        loi |= not ok
        print(f"   trung tâm {c} {TEN_TT[c]:<10} train {a:>5} test {b:>5}   {'khớp README' if ok else f'LỆCH (README {tr}/{te})'}")
    cot = ["image", "target", "center", "trung_tam", "phan", "fold2"]
    chia[[c for c in cot if c in chia.columns]].to_csv(RA / "fed_isic_chia.csv", index=False)
    kt = [Image.open(RA_ANH["cc"] / f"{i}.jpg").size for i in chia.image.iloc[:200] if (RA_ANH["cc"] / f"{i}.jpg").exists()]
    print(f"   cỡ ảnh (200 ảnh đầu): cạnh ngắn {min(min(s) for s in kt)}–{max(min(s) for s in kt)}, "
          f"cạnh dài {min(max(s) for s in kt)}–{max(max(s) for s in kt)}")
    print(f"   phân bố nhãn (target 0–7): {np.bincount(chia.target, minlength=8).tolist()}")
    print(f"\nĐã ghi {RA / 'fed_isic_chia.csv'}" + ("" if not (thieu or loi) else "   *** CÓ LỖI Ở TRÊN — dán lại cho Claude ***"))


if __name__ == "__main__":
    main()
