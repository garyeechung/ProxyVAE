#!/bin/sh
beta1=0.005
beta2_values=(0.001 0.003 0.005 0.007 0.009 0.01 0.1 1)  #  0.007 0.009 0.01
echo "======================================================"
echo "run beta1 values: $beta1"
echo "======================================================"


echo "beta1 = $beta1 Phase 1 Training for CVAE"
CUDA_VISIBLE_DEVICES=1 python -m src.mnist.group_trainers.invarep.main_phase1 --beta1 $beta1 --epochs 500 --batch_size 500 --lr 5e-4

for beta2 in ${beta2_values[@]}; do
    echo "beta1 = $beta1, beta2 = $beta2 Phase 2 Training for ProxyVAE and post-hoc"
    CUDA_VISIBLE_DEVICES=1 python -m src.mnist.group_trainers.invarep.main_phase2 --beta1 $beta1 --beta2 $beta2 --epochs 500 --batch_size 500 --lr 5e-4
done
echo "All runs completed."

# 0.001 0.01 0.03 0.05 0.06 0.07 0.08 0.09 0.1 1.0 10.0 100.0 1000.0