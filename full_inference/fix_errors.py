import re
import os
import subprocess
import csv

STEP_SIZE = 2000

settings = dict(
    GPU="gpu_4_a100",
    TIME="02:00:00",
    JNAME="finf-job",  # adjust
    INPUT="data/works.csv",
    LLAMA_VARIANT="13B-v2",
    MODEL_ID="ft-xxl",
    START=0,  # adjust
    N=0,  # adjust
    BATCH_SIZE=10,
    MAX_NEW_TOKENS=650,
)


SHELL_JOB_FOLDER = "sjobs/"
TEMPLATE = """#!/bin/sh
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

python3 -u full_inference.py \\
 --llama_variant {VARIANT} \\
 --model_id {MODEL_ID} \\
 --start {START} \\
 --n {N} \\
 --input_file {INPUT} \\
 --batch_size {BATCH_SIZE} \\
 --max_new_tokens {MAX_NEW_TOKENS} 
"""


class Slurm:
    def __init__(self, template, folder):
        self.template = template
        self.folder = folder

    def amount_submitted(self):
        return len(self.submitted_jobs())

    def submitted_jobs(self):
        out = Slurm.execute('squeue -o "%j" --noheader')
        return Slurm.keep_valid(out.split("\n"))

    def submit(self, settings):
        os.makedirs(self.folder, exist_ok=True)
        file_name = f"finf-{settings['job_id']}.sh"
        full_path = os.path.join(self.folder, file_name)
        self.create_shell_script(**settings)
        print(f"sbatch {full_path}")
        Slurm.execute(f"sbatch {full_path}")

    def create_shell_script(self, path, **kwargs):
        print(f"Creating shell script: {path}")
        with open(path, "w") as f:
            f.write(self.template.format(**kwargs))

    @staticmethod
    def execute(cmd):
        print(f"Executing: {cmd}")

        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        ).stdout.strip()

    @staticmethod
    def keep_valid(lst):
        return [x for x in lst if x]


regex = re.compile(r"(\d+)_(\d+)")
inf_dir = (
    "/pfs/work7/workspace/scratch/fb6372-matconcepts/data/inference_13B-v2/ft-xxl/"
)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_line_count(path):
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        return sum(1 for _ in csv_reader)


errors = []

for f in os.listdir(inf_dir):
    plain_name = get_filename(inf_dir + f)
    line_count = get_line_count(inf_dir + f)
    mo = regex.search(plain_name)
    start = int(mo.group(1))
    end = int(mo.group(2))

    if line_count != STEP_SIZE:
        errors.append(dict(start=start, end=end, processed=line_count, file=plain_name))


N_DEV_GPU_THRESHOLD = 300

for error in errors:
    start = error["start"] + error["processed"]
    n = error["end"] - start

    settings["GPU"] = "gpu_4_a100" if n > N_DEV_GPU_THRESHOLD else "dev_gpu_4_a100"
    settings["TIME"] = "02:00:00" if n > N_DEV_GPU_THRESHOLD else "00:30:00"

    settings["START"] = start
    settings["N"] = n
    settings["JNAME"] = f"finf-{start}-{error['end']}"
