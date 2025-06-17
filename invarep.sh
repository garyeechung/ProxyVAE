#!/bin/sh

# beta1_values=(0.001 0.01 0.1)
# beta1_values=(1.0 10.0)
beta1_values=(100.0 1000.0)

beta2_values=(0.001 0.01 0.1 1.0 10.0 100.0 1000.0)
echo "======================================================"
echo "run beta1 values: ${beta1_values[@]}"
echo "======================================================"

for beta1 in ${beta1_values[@]}; do
    echo "beta1 = $beta1 Phase 1 Training for CVAE"
    CUDA_VISIBLE_DEVICES=0 python  -m src.mnist.group_trainers.invarep.main_phase1 --beta1 $beta1 --epochs 500 --batch_size 500 --lr 5e-4

    for beta2 in ${beta2_values[@]}; do
        echo "beta1 = $beta1, beta2 = $beta2 Phase 2 Training for ProxyVAE and post-hoc"
        CUDA_VISIBLE_DEVICES=0 python -m src.mnist.group_trainers.invarep.main_phase2 --beta1 $beta1 --beta2 $beta2 --epochs 500 --batch_size 500 --lr 5e-4
    done
done
echo "All runs completed."

