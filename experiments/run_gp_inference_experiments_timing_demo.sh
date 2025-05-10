#!/bin/bash

#SBATCH --job-name=gp_inference_job      # Job name
#SBATCH --output=gp_inference_%j.out     # Standard output and error log
#SBATCH --error=gp_inference_%j.err      # Error log
#SBATCH --time=00:10:00                  # Time limit hrs:min:sec
#SBATCH --partition=gpu                  # Partition to submit to
#SBATCH --gres=gpu:1                     # Request 4 GPUs (adjustable via num_gpus argument)
#SBATCH --constraint=GPU_SKU:L40S        # Ensure use of L40S GPUs
#SBATCH --cpus-per-task=4                # Number of CPU cores per GPU
#SBATCH --nodes=1                        # Number of nodes

# This script is used to run timing experiments on the Stanford Sherlock cluster

# Activate the virtual environment
source /home/groups/udell/gp_inference_env/bin/activate

# Ensure the virtual environment is activated correctly
if [[ -z "$VIRTUAL_ENV" ]] || [[ "$VIRTUAL_ENV" != "/home/groups/udell/gp_inference_env" ]]; then
    echo "Virtual environment not activated correctly. Exiting."
    exit 1
fi

# Ensure the correct Python executable is being used
if [[ "$(which python3)" != "/home/groups/udell/gp_inference_env/bin/python3" ]]; then
    echo "Incorrect Python executable: $(which python3). Exiting."
    exit 1
fi

# Set the PYTHONPATH to the current directory for module resolution
export PYTHONPATH=$(pwd)

# Define datasets
datasets=(3droad)

# Check if the seed and number of GPUs were provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <seed> <num_gpus>"
    exit 1
fi

seed=$1
max_gpus=$2

# Extract the allocated GPU IDs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Exiting."
    exit 1
fi

# Split the GPU IDs into an array
allocated_gpus=( $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ') )

# Run the command for each dataset and each GPU count from 1 to max_gpus
for dataset in "${datasets[@]}"; do
    for num_gpus in $(seq 1 $max_gpus); do
        # Select the correct number of GPUs (create a subset of the allocated GPUs)
        selected_gpus=$(echo ${allocated_gpus[@]:0:$num_gpus} | tr ' ' ',')

        # echo "Running for dataset: $dataset with seed: $seed on devices: $selected_gpus using $num_gpus GPUs"
        python3 experiments/gp_inference_dataset_seed.py --dataset "$dataset" --seed "$seed" --devices "$selected_gpus" --timing
    done
done
