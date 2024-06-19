# FLiP: Towards Comprehensive and Reliable Evaluation of Federated Prompt Learning

## Setup Environments
We use Anaconda to install and manage the required Python packages.
We provide the shell script for seting up the environments, which is `/tools/install_pt_2100.sh`.

## Download datasets
The instructions for downloading and preparing datasets can be found [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

## Run Experiments
By default, we use SLURM for job scheduling in order to achieve large-scale evaluation.
For example, the shell script for running experiments that sweeps the combination of 
various algorithms and datasets is `/scripts/run_batch.sh`.
To run experiments without SLURM, 
turn off the SLURM option in `run_batch.sh` to  `--slurm=false`.