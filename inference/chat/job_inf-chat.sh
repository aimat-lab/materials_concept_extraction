#!/bin/sh
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --job-name=inf13B-v2-chat
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/inf-chat.output
#SBATCH --mail-user=fb6372@partner.kit.edu
module load devel/cuda/11.8
source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u inference-chat.py \
--input_file data/inf-chat.csv \
--llama_variant 13B-v2-chat \
--model_id llama-v2-chat \
--batch_size 10 \
--inf_limit 10 \
--max_new_tokens 1024 \
--use_base_model False \
--prompt_file data/prompt-inference.txt