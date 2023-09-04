# Deprecated: LLaMa 1 Weights

## Obtain LLaMa Weights (inofficial)

From [Github](https://github.com/shawwn/llama-dl):
`curl -o- https://raw.githubusercontent.com/shawwn/llama-dl/56f50b96072f42fb2520b1ad5a1d6ef30351f23c/llama.sh | bash`

## Convert Weights

`python3 conversion_script.py --input_dir ./LlamaW --model_size 13B --output_dir ./llama-13B`

# LLaMa 2 Weights

Can be obtained by filling out Meta's form. You can then download the weights via huggingface directly. No conversion needs to be done.

# Cuda Compatible Installation

For CUDA 11.8:
`pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`

# HPC Commands

File size:
`du -shx *`

Go to Workspace:
`cd $(ws_find matconcepts)`

Available resources at the moment:
`sinfo_t_idle`

Copy files from $HOME to WS:
`cp $HOME/train.py .`
`cp $HOME/inference.py .`

Dispatch jobs:
`sbatch --partition=gpu_4_a100 job_train.sh`
`sbatch --partition=gpu_4_a100 job_inf.sh`

List current jobs:
`squeue`

Detailed information:
`scontrol show job <id>`
