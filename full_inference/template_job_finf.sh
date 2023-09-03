#!/bin/sh
#SBATCH --partition={GPU}
#SBATCH --gres=gpu:1
#SBATCH --time={TIME}
#SBATCH --job-name={JNAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sjobs/{JNAME}.output
#SBATCH --mail-user=fb6372@partner.kit.edu

module load devel/cuda/11.8

source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u full_inference.py \
 --llama_variant {VARIANT} \
 --model_id {MODEL_ID} \
 --start {START} \
 --n {N} \
 --input_file {INPUT} \
 --batch_size {BATCH_SIZE} \
 --max_new_tokens {MAX_NEW_TOKENS} 