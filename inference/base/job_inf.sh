#!/bin/sh
#SBATCH --partition=gpu_4_a100
#SBATCH --job-name=inf-xxl
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/inf%j.output
#SBATCH --mail-user=fb6372@partner.kit.edu
module load devel/cuda/11.8
source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u inference.py \
--input_file data/untagged.csv \
--llama_variant 13B-v2 \
--model_id ft-xxl \
--batch_size 20 \
--inf_limit 100 \
--max_new_tokens 650