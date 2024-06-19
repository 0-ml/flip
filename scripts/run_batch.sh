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
--np_forward_times=8 \
--eval_multi=false \
--client_eval=false \
--slurm=true \
--verbose2 > outputs/"${algo}_${dataset}.out" &
sleep 5
done
done
wait
echo "All jobs done!"
