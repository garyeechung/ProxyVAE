#!/bin/bash

beta1_values=(0.0 0.01 1.0)
beta2_values=(0.0 0.01 1.0)

for beta1 in "${beta1_values[@]}"; do
    for beta2 in "${beta2_values[@]}"; do
        beta1_str=$(printf "%.1E" $beta1)
        beta2_str=$(printf "%.1E" $beta2)

        src="/home/chungk1/Repositories/InvaRep/checkpoints/adni/resnet18_backup/invarep/beta1_${beta1_str}/beta2_${beta2_str}/proxy2invarep_best.pth"

        # New unique target directory per (beta1, beta2) pair
        dst_dir="/home/chungk1/Repositories/InvaRep/checkpoints/adni/resnet18/invarep/beta1_${beta1_str}/beta2_${beta2_str}"
        dst="${dst_dir}/proxy2invarep_best.pth"

        mkdir -p "$dst_dir"

        if [ -f "$src" ]; then
            echo "Copying: $src -> $dst"
            cp "$src" "$dst"
        else
            echo "File not found: $src"
        fi
    done
done
