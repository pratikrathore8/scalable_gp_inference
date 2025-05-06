#!/bin/bash

# Define datasets and devices
datasets=(taxi)
devices="2 3 4"

# Check if seed was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

seed=$1

# Run the command for each dataset with the provided seed
for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset with seed: $seed"
    python experiments/gp_inference_dataset_seed.py --dataset "$dataset" --seed "$seed" --devices $devices
done
