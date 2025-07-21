#!/bin/sh
device=0
beta1_values=(0.0 0.0001 1.0)  # 0.0 0.0001 1.0
beta2_values=(0.0 0.0001 1.0)  # 0.0 0.0001 1.0
modality="t1"
backbone="resnet18"
bound_z_by="tanh"
batch_size=256
batch_per_epoch=10
lr_phase1=1e-4
lr_phase2=1e-5
epochs_phase1=5000
epochs_phase2=500
echo "======================================================"
echo "run beta1 values: ${beta1_values[@]}"
echo "======================================================"

for beta1 in ${beta1_values[@]}; do
    echo "beta1 = $beta1 Phase 1 Training"
    nohup env CUDA_VISIBLE_DEVICES=$device python -m src.cnn.trainers.adni.invarep.main_phase1 --beta1 $beta1 --epochs $epochs_phase1 --batch_size $batch_size --bootstrap --lr $lr_phase1 --backbone $backbone --bound_z_by $bound_z_by --modality $modality --batch_per_epoch $batch_per_epoch > logs/$modality/$backbone-$bound_z_by/phase1_beta1_$(printf "%.1E" $beta1).log 2>&1

    for beta2 in ${beta2_values[@]}; do
        echo "beta1 = $beta1, beta2 = $beta2 Phase 2 Training"
        nohup env CUDA_VISIBLE_DEVICES=$device python -m src.cnn.trainers.adni.invarep.main_phase2 --beta1 $beta1 --beta2 $beta2 --epochs $epochs_phase2 --batch_size $batch_size --bootstrap --lr $lr_phase2 --backbone $backbone --bound_z_by $bound_z_by --modality $modality --batch_per_epoch $batch_per_epoch > logs/$modality/$backbone-$bound_z_by/phase2_beta1_$(printf "%.1E" $beta1)_beta2_$(printf "%.1E" $beta2).log 2>&1
    done

done
echo "All runs completed."
