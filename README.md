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
We provide the shell script for setting up the environments, which is `/tools/install_pt_2100.sh` shown as follows:

```
conda create -n pt-2100 python=3.10.14 -y
source ${HOME}/app/anaconda3/bin/activate pt-2100   # need customization
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y matplotlib pandas
conda install -y tensorboard tensorboardx
conda install -y tqdm scikit-learn termcolor

conda install -y numpy==1.23.5
pip install h5py ftfy regex
```
Notably, the default installation path of anaconda is `${HOME}/app/anaconda3`,
which may requires customization to activate the created conda env.


# Download Datasets  
The instructions for downloading and preparing datasets can be found [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

# Run Experiments
By default, we use SLURM for job scheduling in order to achieve large-scale evaluation.
For example, the shell script for running experiments that sweeps the combination of 
various algorithms and datasets is `/scripts/run_batch.sh`.
To run experiments without SLURM, 
turn off the SLURM option in `run_batch.sh` to  `--slurm=false`.

```
cd ../
mkdir outputs
prefix="srun --exclusive -n 1 -c 8 --gpus-per-task=1 --mem=15G"

for algo in CLIP CoOp CoCoOp PromptSRC OTP KgCoOp PLOT ProDA ProGrad; do
for dataset in caltech101 fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf dtd; do
$prefix python main.py \
--times=3 \
--benchmark=base2novel \
--data_root=~/data/prompt \
--num_workers=6 \
--precision=amp \
--dataset=$dataset \
--image_backbone=RN50 \
--prompt_algo=$algo \
--optim_name=sgd \
--lr_scheduler='cos' \
--split_alpha=0.1 \
--loss_type=ce \
--central=false \
--num_clients=10 \
--num_shot=8 \
--optim_momentum=0.9 \
--local_learning_rate=0.002 \
--batch_size=16 \
--eval_scaler=2 \
--num_prompt=1 \
--local_epochs=1 \
--global_rounds=50 \
--prompt_batch_size=2 \
--eval_multi=false \
--client_eval=false \
--slurm=true \
--verbose2 > outputs/"${algo}_${dataset}.out" &
sleep 5
done
done
wait
echo "All jobs done!"
```

# Check Results
Each experimental run will create a `results` folder to save benchmarked metrics,
an `output` folder to keep training logs and a `summaries` folder to store
tensorboard logs for visualization.