#!/bin/sh
beta_values=(0.0 0.0001 1.0)  # 0.0 0.0001 1.0
device=1
backbone="resnet18"
bound_z_by="tanh"
batch_size=10000
lr=1e-5
epochs=500
echo "======================================================"
echo "run beta values: ${beta_values[@]}"
echo "======================================================"

for beta in ${beta_values[@]}; do
    echo "beta = $beta info bottleneck training"
    nohup env CUDA_VISIBLE_DEVICES=$device python -m src.cnn.trainers.cifar100.infobottleneck --beta $beta --epochs $epochs --batch_size $batch_size --lr $lr --backbone $backbone --bound_z_by $bound_z_by > logs/cifar100/$backbone-$bound_z_by/ib_beta_$(printf "%.1E" $beta).log 2>&1

done
echo "All runs completed."
