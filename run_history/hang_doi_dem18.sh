exec 9>/tmp/ham_fl_gpu.lock; flock 9
cd ~/ham_fl && source .venv/bin/activate
for bb in resnet18 deit_tiny_patch16_224 efficientnet_b3 resnet50.a1h_in1k vit_small_patch16_224.augreg_in21k_ft_in1k convnext_tiny.fb_in22k_ft_in1k; do
  for lr in 4 1; do
    python dp_tuyen_tinh.py --backbone $bb --tien-xu-ly p1 --luoi --cac-kich-ban nguon --lr $lr --ra ketqua/dp_tuyen_tinh_lr$lr
  done
done
python phan_tich_dac_trung.py > dem18_dac_trung.txt 2>&1; cat dem18_dac_trung.txt
python sim_may_chu.py > dem18_may_chu.txt 2>&1; tail -40 dem18_may_chu.txt
echo "=== XONG ĐÊM 18 $(date) ==="
python bao_dong.py "Đêm 18 xong" "$(head -14 dem18_dac_trung.txt)"
