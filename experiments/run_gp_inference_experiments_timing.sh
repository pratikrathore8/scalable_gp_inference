#!/bin/bash

#SBATCH --job-name=gp_inference_job      # Job name
#SBATCH --output=gp_inference_%j.out     # Standard output and error log
#SBATCH --error=gp_inference_%j.err      # Error log
#SBATCH --time=4:00:00                  # Time limit hrs:min:sec
#SBATCH --partition=gpu                  # Partition to submit to
#SBATCH --gres=gpu:4                     # Request 4 GPUs (adjustable via num_gpus argument)
#SBATCH --constraint=GPU_SKU:V100_PCIE   # Ensure use of V100-PCIE GPUs
#SBATCH --cpus-per-task=8                # Number of CPU cores per GPU
#SBATCH --nodes=1                        # Number of nodes

# This script is used to run timing experiments on the Stanford Sherlock cluster

# Activate the virtual environment
source /home/users/pratikr/gp_inference_env/bin/activate

# Define datasets
datasets=(acsincome yolanda malonaldehyde benzene 3droad song houseelec)

# Check if the seed and number of GPUs were provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <seed> <num_gpus>"
    exit 1
fi

seed=$1
num_gpus=$2

# Extract the allocated GPU IDs
allocated_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')

# Select the correct number of GPUs
selected_gpus=$(echo $allocated_gpus | cut -d' ' -f1-$num_gpus)

# Run the command for each dataset with the provided seed and number of GPUs
for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset with seed: $seed on devices: $selected_gpus"
    python3 experiments/gp_inference_dataset_seed.py --dataset "$dataset" --seed "$seed" --devices "$selected_gpus" --timing
done
