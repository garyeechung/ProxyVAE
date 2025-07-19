#!/bin/sh
beta_values=(0.0 0.0001 0.01 1.0)
device=1
echo "======================================================"
echo "run beta values: ${beta_values[@]}"
echo "======================================================"

for beta in ${beta_values[@]}; do
    echo "beta = $beta info bottleneck training"
    nohup env CUDA_VISIBLE_DEVICES=$device python -m src.cnn.trainers.adni.infobottleneck.main --beta $beta --epochs 1000 --batch_size 256 --lr 1e-5 --backbone resnet18 > logs/resnet18/ib_beta_$(printf "%.1E" $beta).log 2>&1

done
echo "All runs completed."
