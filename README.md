## Preparation with High Performance Computing (HPC)

- [x] Discuss the 5 annotated examples with Pascal
- [x] Annote 100 abstracts with concepts
      ~~- [ ] Get access to weights directory (Felix)~~
- [x] Fine-tune the model (Script)
      ~~- [ ] Generate concepts for 300 more abstracts~~
      ~~- [ ] Correct 300 concepts~~
      ~~- [ ] Fine-tune on these 300 abstracts~~
- [x] Run on all abstracts. Output: data/raw/
- [ ] Analyze concepts

## Obtain LLaMa Weights (inofficial)

From [Github](https://github.com/shawwn/llama-dl):
`curl -o- https://raw.githubusercontent.com/shawwn/llama-dl/56f50b96072f42fb2520b1ad5a1d6ef30351f23c/llama.sh | bash`

## Cuda Compatible Installation

For CUDA 11.8:
`pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`

## Workspace: Matconcepts

Path:
`/pfs/work7/workspace/scratch/fb6372-matconcepts`

Copy files:
`scp file.txt fb6372@bwunicluster.scc.kit.edu:/pfs/work7/workspace/scratch/fb6372-matconcepts`
`ctc file.txt`
`scp file.txt $WS`
`scp $WS/file.txt .`

Download all inference files:
`scp $WS/data/inference_13B/full-finetune-100/\* ./data/raw/`

## Convert Weights

`python3 conversion_script.py --input_dir ./LlamaW --model_size 13B --output_dir ./llama-13B`

## Commands

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
