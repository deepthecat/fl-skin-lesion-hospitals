"""
dua_len_wandb.py — đưa TOÀN BỘ các lượt chạy cũ (trước khi có wandb) lên wandb, để có một bảng nhật ký đầy đủ.

    python dua_len_wandb.py            # đưa các lượt chưa đưa
    python dua_len_wandb.py --thu      # chỉ liệt kê sẽ đưa gì, không gửi

Nguồn: mọi file ketqua/<thư mục>/*.json do các script huấn luyện ghi (cấu hình, đường cong từng epoch/vòng,
kết quả val/test, mã băm code). Mỗi file -> một run, gắn nhãn "nhap_lai" để phân biệt với lượt ghi trực tiếp.
Thêm một run "tong_hop_phan_tich": các bảng .csv phân tích, log chạy đêm (dem*.txt), BAN_DO.md, SO_TAY_KHAI_NIEM.md.

Hạn chế cần nói rõ khi báo cáo: lượt cũ chỉ có MÃ BĂM code (10 ký tự SHA-256), không có bản code đầy đủ của lúc đó,
và không có số liệu hệ thống (GPU, thời gian từng bước). Từ nay mọi lượt mới ghi đủ, kèm commit git.
Đã đưa lượt nào thì ghi vào ketqua/.wandb_da_dua.txt, chạy lại không bị trùng.
"""
import argparse
import json
import os
from pathlib import Path

from du_lieu import RA
from nhat_ky import DU_AN, co_khoa, nhom_cua, phang

DANH_DAU = RA / ".wandb_da_dua.txt"
GOC = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu", action="store_true")
    arg = ap.parse_args()
    da = set(DANH_DAU.read_text().split()) if DANH_DAU.exists() else set()
    tep = sorted(f for f in RA.glob("*/*.json") if not f.name.startswith("."))
    can = [f for f in tep if f"{f.parent.name}/{f.stem}" not in da]
    print(f"{len(tep)} file kết quả, {len(can)} chưa đưa lên.")
    theo_loai = {}
    for f in can:
        theo_loai[f.parent.name] = theo_loai.get(f.parent.name, 0) + 1
    for k, v in sorted(theo_loai.items()):
        print(f"  {k:<16}{v:>4}")
    if arg.thu or not can:
        return

    import wandb
    che_do = os.environ.get("WANDB_MODE") or ("online" if co_khoa(wandb) else "offline")
    print(f"Chế độ: {che_do}" + ("  (đẩy lên sau: wandb sync wandb/offline-run-*)" if che_do == "offline" else ""))
    for i, f in enumerate(can, 1):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  bỏ qua {f.name}: {e}"); continue
        ten = d.get("ten", f.stem)
        cfg = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
               for k, v in (d.get("cau_hinh") or {}).items()}
        cfg.update(loai=f.parent.name, ma_script=d.get("ma_script"), ma_du_lieu=d.get("ma_du_lieu"), nhap_lai=True)
        run = wandb.init(project=DU_AN, name=ten, id=("cu-" + f.parent.name + "-" + ten)[:120].replace("/", "-"),
                         group=nhom_cua(ten), job_type=f.parent.name, config=cfg, tags=["nhap_lai", f.parent.name],
                         dir=str(GOC), mode=che_do, reinit=True)
        for h in d.get("lich_su") or []:
            buoc = h.get("vong", h.get("epoch"))
            run.log(phang(h), step=int(buoc)) if buoc is not None else run.log(phang(h))
        for k, v in phang(dict(ket_qua=d.get("ket_qua", {}), giay=d.get("giay", 0))).items():
            run.summary[k] = v
        run.save(str(f), base_path=str(RA), policy="now")
        run.finish()
        with open(DANH_DAU, "a") as g:
            g.write(f"{f.parent.name}/{f.stem}\n")
        print(f"  [{i}/{len(can)}] {f.parent.name}/{ten}")

    run = wandb.init(project=DU_AN, name="tong_hop_phan_tich", job_type="phan_tich", tags=["phan_tich"],
                     dir=str(GOC), mode=che_do, reinit=True)
    for c in sorted(RA.glob("*.csv")):
        try:
            import pandas as pd
            run.log({f"bang/{c.stem}": wandb.Table(dataframe=pd.read_csv(c))})
        except Exception as e:
            print(f"  bảng {c.name}: {e}")
    for t in list(GOC.glob("dem*.txt")) + [GOC / "BAN_DO.md", GOC / "SO_TAY_KHAI_NIEM.md"]:
        if t.exists():
            run.save(str(t), base_path=str(GOC), policy="now")
    run.finish()
    print("Xong. Mở https://wandb.ai -> project", DU_AN)


if __name__ == "__main__":
    main()
