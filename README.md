# FLiP: Towards Comprehensive and Reliable Evaluation of Federated Prompt Learning
<h1 align="center">
    <img src='./assets/FLiP_logo.png'>
</h1>

# Brief Introduction  
FLiP is a comprehensive benchmark suite for large-scale evaluation of federated prompt learning methods, particularly on vision tasks. 
It integrates a rich set of federated prompt learning algorithms to provide a reproducible benchmark with aligned experimental settings to track the cutting-edge advancements in this field. 
FLiP also offers off-the-shelf highly-reusable functionalities for flexible customization and agile developing of novel federated prompt learning algorithms.
Blessed by the decoupled design of federated learning and prompt learning modules,
it can be readily extended to harvest the progress from federated learning and prompt learning communities.
FLiP is also remarkably more convenient to use than modifying centralized prompt learning repos to adapt to federated settings.

# High-level Overview of Code Structure  
```
FLip
├── flcore
│   ├── clients                  # local training algorithms for clients
│   │   ├── client_base.py
│   │   ├── client_fedavg.py
│   │   └── client_fedotp.py
│   ├── datasets                 # dataset partition and processing
│   │   ├── base.py
│   │   ├── caltech101.py
│   │   ├── domain_net.py
│   │   ├── imagenet.py
│   │   ├── ...
│   │   ├── imageloader.py
│   │   ├── info.py
│   │   ├── randaugment.py
│   │   ├── randtransform.py
│   │   └── utils.py
│   ├── models                   # model configurations
│   │   ├── clip                 # CLIP models and prompt learning algorithms
│   │   ├── cnn
│   │   ├── __init__.py
│   │   ├── text
│   │   └── utils.py
│   ├── optimizers               # customizable optimizers
│   │   ├── __init__.py
│   │   ├── sam.py
│   │   └── utils.py
│   ├── pretty
│   │   ├── history.py
│   │   └── logger.py
│   ├── servers                  # global aggregation algorithms on server
│   │   ├── ground_metric.py
│   │   ├── server_base.py
│   │   ├── server_fedavg.py
│   │   └── server_fedotp.py
│   └── utils.py
├── main.py
├── scripts                      # shell scripts for reproducing results
│   └── run_batch.sh
└── tools                        # misc tools for building envs, dataset processing, etc
    └── install_pt_2100.sh
```

# Setup Environments  
We use Anaconda to install and manage the required Python packages.
We provide the shell script for setting up the environments, which is `/tools/install_pt_2100.sh`.

# Download Datasets  
The instructions for downloading and preparing datasets can be found [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

# Run Experiments
By default, we use SLURM for job scheduling in order to achieve large-scale evaluation.
For example, the shell script for running experiments that sweeps the combination of 
various algorithms and datasets is `/scripts/run_batch.sh`.
To run experiments without SLURM, 
turn off the SLURM option in `run_batch.sh` to  `--slurm=false`.

# Check Results
Each experimental run will create a `results` folder to save benchmarked metrics,
an `output` folder to keep training logs and a `summaries` folder to store
tensorboard logs for visualization.