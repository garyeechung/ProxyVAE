#!/bin/sh

for beta in 0.001 0.01 0.1 1.0 10.0 100.0 1000.0; do
    echo "Running with beta = $beta"
    CUDA_VISIBLE_DEVICES=1 python  -m src.mnist.group_trainers.infobottleneck.main --beta $beta --epochs 300 --batch_size 500 --lr 5e-4 --device cpu
    echo "Finished beta = $beta"
done
echo "All runs completed."
