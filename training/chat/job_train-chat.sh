#!/bin/sh
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --job-name=train13B_v2-chat
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=models/train13B_v2-chat.output
#SBATCH --mail-user=fb6372@partner.kit.edu

module load devel/cuda/11.8
source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u train-chat.py \
 --llama_variant 13B-v2-chat \
 --output_dir llama-v2-chat \
 --num_epochs 4 \
 --lr 5e-4 \
 --size_train_dataset 1 \
 --prompt_file data/prompt-training.txt