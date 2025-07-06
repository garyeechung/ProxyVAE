#!/bin/sh
beta1_values=(0.0000000001 0.01 0.1 0.3 0.5 1.0)
echo "======================================================"
echo "run beta1 values: ${beta1_values[@]}"
echo "======================================================"

for beta1 in ${beta1_values[@]}; do
    echo "beta1 = $beta1 Phase 1 Training for CVAE"
    CUDA_VISIBLE_DEVICES=2 python -m src.adni.trainers.invarep.main_phase1 --beta1 $beta1 --epochs 500 --batch_size 500 --lr 5e-4
done
echo "All runs completed."
