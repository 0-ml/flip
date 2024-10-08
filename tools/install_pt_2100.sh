conda create -n pt-2100 python=3.10.14 -y
source ${HOME}/app/anaconda3/bin/activate pt-2100
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y matplotlib pandas
conda install -y tensorboard tensorboardx
conda install -y tqdm scikit-learn termcolor

conda install -y numpy==1.23.5
pip install h5py ftfy regex
