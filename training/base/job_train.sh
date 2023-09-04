#!/bin/sh
#SBATCH --partition=gpu_4_a100
#SBATCH --job-name=train-xxl
#SBATCH --time=01:20:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=models/train13B_v2-xxl.output
#SBATCH --mail-user=fb6372@partner.kit.edu

module load devel/cuda/11.8
source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u train.py \
 --train_file data/train_xxl.csv \
 --llama_variant 13B-v2 \
 --output_dir ft-xxl \
 --num_epochs 4 \
 --lr 5e-4 \
 --size_train_dataset 1