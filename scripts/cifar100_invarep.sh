#!/bin/sh
device=2
beta1_values=(0.0)  # 0.0 0.0001 1.0
beta2_values=(1.0)  # 0.0 0.0001 1.0

backbone="resnet18"
bound_z_by="tanh"
batch_size=10000

lr_phase1=1e-4
lr_phase2=1e-5
epochs_phase1=5000
epochs_phase2=500
echo "======================================================"
echo "run beta1 values: ${beta1_values[@]}"
echo "======================================================"

for beta1 in ${beta1_values[@]}; do
    # echo "beta1 = $beta1 Phase 1 Training"
    log_path="logs/cifar100/$backbone-$bound_z_by/phase1_beta1_$(printf "%.1E" $beta1).log"
    mkdir -p $(dirname $log_path)
    # nohup env CUDA_VISIBLE_DEVICES=$device python -m src.cnn.trainers.cifar100.main_phase1 --beta1 $beta1 --epochs $epochs_phase1 --batch_size $batch_size --lr $lr_phase1 --backbone $backbone --bound_z_by $bound_z_by > $log_path 2>&1

    for beta2 in ${beta2_values[@]}; do
        echo "beta1 = $beta1, beta2 = $beta2 Phase 2 Training"
        nohup env CUDA_VISIBLE_DEVICES=$device python -m src.cnn.trainers.cifar100.main_phase2 --beta1 $beta1 --beta2 $beta2 --epochs $epochs_phase2 --batch_size $batch_size --lr $lr_phase2 --backbone $backbone --bound_z_by $bound_z_by > logs/cifar100/$backbone-$bound_z_by/phase2_beta1_$(printf "%.1E" $beta1)_beta2_$(printf "%.1E" $beta2).log 2>&1
    done

done
echo "All runs completed."
