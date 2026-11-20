"""
du_lieu.py — quy ước dùng chung. Mọi notebook và script import từ đây.

Lý do tồn tại: đề tài trước có hai lỗi (ngưỡng mặt nạ, thứ tự khung) đều sinh ra từ chuyện
mỗi script tự viết lại một kiểu đọc dữ liệu. Ở đây mỗi quy ước chỉ được viết đúng một chỗ.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

GOC = Path(os.environ.get("HAM_GOC", Path.home() / "ham_fl" / "data"))
RA = GOC.parent / "ketqua"
LOP = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
TB = np.array([0.485, 0.456, 0.406], np.float32)   # chuẩn hoá ImageNet
DL = np.array([0.229, 0.224, 0.225], np.float32)


def doc_chia_tap():
    """Đọc ketqua/chia_tap.csv (do notebook 02 sinh ra)."""
    f = RA / "chia_tap.csv"
    if not f.exists():
        raise SystemExit("Chưa có ketqua/chia_tap.csv — chạy notebook 02 trước.")
    return pd.read_csv(f)


TIEN_XU_LY = {
    "p0": "ép thẳng về s×s (méo tỉ lệ 4:3) — pipeline thô ban đầu",
    "p1": "giữ tỉ lệ: cạnh ngắn về s rồi cắt giữa s×s",
    "p2": "p1 + chuẩn hoá màu Shades of Gray (p=6)",
}


def shades_of_gray(a, p=6):
    """Chuẩn hoá màu Shades of Gray (Finlayson & Trezzi 2004). Ước lượng màu ánh sáng chiếu vào ảnh
    bằng trung bình bậc p của từng kênh, rồi chia đi để ánh sáng thành trung tính. Ảnh vốn trung
    tính thì không đổi. Mỗi ảnh tự xử lý một mình, không cần thống kê của ảnh khác -> không tốn
    gì về riêng tư khi dùng trong federated."""
    x = a.astype(np.float32)
    e = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    e = e / max(np.linalg.norm(e), 1e-6)
    return np.clip(x / (e * np.sqrt(3)), 0, 255).astype(np.uint8)


def doc_anh(p, s, kieu="p0"):
    """Ảnh RGB uint8 s×s. draft() cho JPEG giải nén ở độ phân giải thấp -> nhanh gấp nhiều lần."""
    with Image.open(p) as im:
        im.draft("RGB", (s * 2, s * 2))
        im = im.convert("RGB")
        if kieu == "p0":
            return np.asarray(im.resize((s, s), Image.BILINEAR))
        w, h = im.size
        k = s / min(w, h)
        im = im.resize((max(s, round(w * k)), max(s, round(h * k))), Image.BILINEAR)
        w, h = im.size
        x0, y0 = (w - s) // 2, (h - s) // 2
        a = np.asarray(im.crop((x0, y0, x0 + s, y0 + s)))
    if kieu == "p2":
        a = shades_of_gray(a)
    return a


def nap_dem(ds, s, kieu="p0"):
    """Toàn bộ ảnh ở kích thước s, dạng mảng uint8 [N, s, s, 3], đúng thứ tự ds.image.
    Mỗi kiểu tiền xử lý có một bộ đệm riêng. Lần đầu giải nén và lưu ra đĩa (~1,5 GB ở 224);
    các lần sau nạp trong vài giây. Kiểu p0 giữ tên file cũ để dùng lại bộ đệm đã có."""
    if kieu not in TIEN_XU_LY:
        raise ValueError(f"kiểu tiền xử lý không biết: {kieu}")
    RA.mkdir(exist_ok=True)
    duoi = "" if kieu == "p0" else f"_{kieu}"
    f = RA / f"dem_{s}{duoi}.npy"
    fid = RA / f"dem_{s}{duoi}_ids.npy"
    if f.exists() and fid.exists():
        ids = np.load(fid)
        vt = {k: i for i, k in enumerate(ids)}
        if all(i in vt for i in ds.image):
            X = np.load(f, mmap_mode="r")
            return np.ascontiguousarray(X[[vt[i] for i in ds.image]])
    print(f"Dựng bộ đệm ảnh {s}x{s} kiểu {kieu} cho {len(ds)} ảnh (chỉ làm một lần)...", flush=True)
    X = np.stack([doc_anh(p, s, kieu) for p in ds.duong_dan])
    np.save(f, X)
    np.save(fid, np.asarray(ds.image.tolist(), dtype=np.str_))
    return X


def cham(y, logit):
    """Các chỉ số. bacc (balanced accuracy) là chỉ số CHÍNH — chỉ số chính thức của ISIC 2018."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    logit = np.asarray(logit, np.float64)
    p = np.exp(logit - logit.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    yh = p.argmax(1)
    d = dict(bacc=balanced_accuracy_score(y, yh), f1_macro=f1_score(y, yh, average="macro"),
             acc=accuracy_score(y, yh))
    try:
        d["auc_macro"] = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(len(LOP))))
    except ValueError:
        d["auc_macro"] = float("nan")
    return {k: float(v) for k, v in d.items()}


def tao_mo_hinh(ten, pretrained=True):
    import timm
    return timm.create_model(ten, pretrained=pretrained, num_classes=len(LOP))


def ma_file(p):
    """10 ký tự đầu SHA-256 của file — ghi vào mọi kết quả để biết nó do phiên bản code nào sinh ra."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:10]
