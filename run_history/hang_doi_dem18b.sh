exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
python sim_may_chu.py > dem18b_may_chu.txt 2>&1
echo "=== XONG ĐÊM 18b $(date) ==="
