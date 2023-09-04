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

# Errors

Sometimes, the job will crash due to limited memory (some batches might exceed the free memory). The data is saved at the end of each batch, therefore, progress will be safe.

Usually, only a tiny fraction of jobs will not complete. To restart these jobs, the start point should be adjusted. This is done by the `fix_errors.py` script.

Set the params in the script according to your setup (e.g. `VARIANT`, `MODEL`, `STEP_SIZE`, `INPUT`). The `BATCH_SIZE` param plays a critical role. If a few jobs didn't complete with `batch_size = 20`, you should set the batch size to 10 for example. If some jobs didn't complete with `batch_size = 10`, you should set the batch size to something even less, for example 5.

In the end, the additionally created `.csv` files should be merged with the data from the full inference run. Adapt and run the `merge.py` script to do so.

# Processing the results

The extracted concepts need to be processed:

- All translated to lower case
- All "/" and "-" replaced by whitespaces
- 'aluminum' replaced by 'aluminium'
- Single element symbols should be replaced by their name (e.g. 'Cu' -> 'copper')
- Duplicate concepts should be removed

The concepts should then be merged with the original data, adding a separate column called `llama_concepts`.

All this can be done by running the `process.py` script.

The resulting file (`materials-science.elements.works.llama-v2.csv`) can then be transferred to the main repo.
