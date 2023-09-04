# Repo Structure

## `training` Folder

Fine-tune LLaMa models to extract concepts. This folder contains the scripts to fine-tune the base and chat variant of LLaMa.

## `tagging` Folder

The data to fine-tune your model must come from somewhere. This folder contains utilities to help with tagging abstracts with their corresponding concepts.

## `inference` Folder

Once you have a fine-tuned model, you can use it to extract concepts from abstracts. This folder contains scripts to run inference on a fine-tuned model.

> Do only use these scripts to process small (<= 1000) amounts of abstracts. For larger amounts, take a look at the `full_inference` folder.

## `fine_tuning` Folder

Improve your model by extending the training data. Run inference of the model on e.g. 100 abstracts and correct the predictions (this should take way less time than tagging the abstracts from scratch). Then fine-tune the model with the old + new data.

## `full_inference` Folder

Once your model is fine-tuned on a sufficiently large dataset, you can use it to extract concepts from a large amount of abstracts. This folder provides scripts to handle large amount of data.

Highlights:

- Data is processed into chunks of {STEP_SIZE}, each chunk is processed by a single job.
- Generated concepts are saved periodically during job execution.
- A tiny scheduler for BWUniCluster allows to monitor the job execution and to keep the queue full until all data is processed.
- If jobs fail, new jobs can be started continuing where the failed job left off. These corrections are saved in a separate file and can be merged with the original concept files.
- In the end, all concepts can be cleaned and merged into a single file.

# General Stuff

## Deprecated: LLaMa 1 Weights

### Obtain LLaMa Weights (inofficial)

From [Github](https://github.com/shawwn/llama-dl):
`curl -o- https://raw.githubusercontent.com/shawwn/llama-dl/56f50b96072f42fb2520b1ad5a1d6ef30351f23c/llama.sh | bash`

### Convert Weights

`python3 conversion_script.py --input_dir ./LlamaW --model_size 13B --output_dir ./llama-13B`

## LLaMa 2 Weights

Can be obtained by filling out Meta's form. You can then download the weights via huggingface directly. No conversion needs to be done.

## Cuda Compatible Installation

For CUDA 11.8:
`pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`

## HPC Commands

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
