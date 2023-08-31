# Using scheduler to run full inference

The inference is run in separate blocks of {STEP_SIZE}.

Adjust the {VARIANT} (e.g. 13B-v2) and {MODEL} to select the desired model.

# Determine the needed amount of time

It is recommended to run the inference on a small subset of the data first to determine the needed amount of time for {STEP_SIZE} samples.

# Preparation

Create `jobs.csv` at top level of repo. Create `sjobs` folder at top level of repo, this folder will contain the logs and
starting scripts for each block of samples. If a job couldn't complete, these scripts can be used later to restart the
extraction (with the option to edit the script to change the parameters).

Run `scheduler.py` inside a `tmux` environment to start the scheduling.

Create input file `works.csv` in the `data/` folder. This file must contain at least `[id, abstract]` columns.
