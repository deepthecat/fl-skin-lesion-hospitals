#!/usr/bin/env bash
# Đêm 16a (chạy NGAY, ngắn ~20–40 phút): (1) đọc kết quả 50 vòng của đêm 15; (2) DP lớp cuối (L0, xương sống đóng băng)
# cho ConvNeXt-atto + EfficientNet-B0, nguồn thật, P1, 6 mức riêng tư × 3 seed — điểm đầu tiên của đường đánh đổi bước 3.
set -u
cd ~/ham_fl
source .venv/bin/activate
for f in dp_tuyen_tinh.py phan_tich_dp_sau.py rmia.py; do [ -f "$f" ] || { echo "LỖI: thiếu $f"; exit 1; }; done
pgrep -f "hang_doi_dem1[0-9]" >/dev/null && { echo "LỖI: đang có hàng đợi khác chạy."; exit 1; }
echo "=== 1. ConvNeXt FedAvg: 50 vòng so với 30 vòng (luật: hơn ở cả 3 seed VÀ TB > 0,02 trên val)"
python - <<'PY'
import json, numpy as np
from du_lieu import RA
def v(t, k):
    f = RA / "lien_ket" / f"fedavg_nguon_convnext_atto_txlp1{t}_kt224_s{k}.json"
    d = json.load(open(f)) if f.exists() else None
    return (d["ket_qua"]["val"]["bacc"], d["ket_qua"]["test"]["bacc"],
            int(np.argmax([r["val_bacc"] for r in d["lich_su"]])) + 1) if d else (np.nan, np.nan, 0)
a = [v("", s) for s in range(3)]; b = [v("_v50", s) for s in range(3)]
for s in range(3):
    print(f"   seed {s}: 30 vòng val {a[s][0]:.3f} test {a[s][1]:.3f} (đỉnh vòng {a[s][2]}) | 50 vòng val {b[s][0]:.3f} test {b[s][1]:.3f} (đỉnh vòng {b[s][2]})")
h = np.array([b[s][0] - a[s][0] for s in range(3)])
print(f"   chênh val TB {h.mean():+.3f}; cả 3 seed dương: {bool(np.all(h > 0))} -> "
      + ("ĐẠT: cân nhắc đổi sang 50 vòng" if np.all(h > 0) and h.mean() > 0.02 else "KHÔNG ĐẠT: giữ 30 vòng"))
PY
export HAMFL_GIA_THUYET="thiet_ke_B3,cong_A"
export HAMFL_DU_DOAN="DP lớp cuối, xương sống đóng băng: ConvNeXt (đặc trưng ImageNet tốt hơn) cao hơn EfficientNet ở mọi ε"
cat > hang_doi_dem16a.sh <<'Q'
exec 9>/tmp/ham_fl_gpu.lock; flock -n 9 || { echo "Đang có hàng đợi khác chạy — thoát."; exit 1; }
cd ~/ham_fl && source .venv/bin/activate
for bb in convnext_atto efficientnet_b0; do
  python dp_tuyen_tinh.py --backbone $bb --tien-xu-ly p1 --luoi --cac-kich-ban nguon
done
echo "=== XONG ĐÊM 16a $(date) ==="
python phan_tich_dp_sau.py
Q
setsid nohup bash hang_doi_dem16a.sh > dem16a.txt 2>&1 &
sleep 60; echo "--- 60 giây đầu:"; grep -v Warning dem16a.txt | tail -4
