# FLiP: Towards Comprehensive and Reliable Evaluation of Federated Prompt Learning

## Setup Environments
We use Anaconda to install and manage the required Python packages.
We provide the shell script for setting up the environments, which is `/tools/install_pt_2100.sh`.

## Download Datasets
The instructions for downloading and preparing datasets can be found [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

## Run Experiments
By default, we use SLURM for job scheduling in order to achieve large-scale evaluation.
For example, the shell script for running experiments that sweeps the combination of 
various algorithms and datasets is `/scripts/run_batch.sh`.
To run experiments without SLURM, 
turn off the SLURM option in `run_batch.sh` to  `--slurm=false`.

## Check Results
Each experimental run will create a `results` folder to save benchmarked metrics,
an `output` folder to keep training logs and a `summaries` folder to store
tensorboard logs for visualization.