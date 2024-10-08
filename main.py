import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import json
import random



from flcore.servers import get_server
from flcore.pretty.logger import log


def parse_results(args):
    exp_name = os.path.join(args.dataset, args.image_backbone, args.fed_algo, args.prompt_algo)
    results_dir = os.path.join('results', exp_name, f'shot_{args.num_shot}')
    json_files = [os.path.join(results_dir, file) for file in os.listdir(results_dir) if 'exps' in file]

    with open(json_files[0], 'r') as json_file:
            json_data = json.load(json_file)
            accs = {k: [] for k in json_data.keys()}

    results = copy.deepcopy(accs)
    for file in json_files:
        with open(file) as json_file:
            json_data = json.load(json_file)
            for k,v in json_data.items():
                accs[k].append(v)
    for k,v in accs.items():
        results[k].append(np.mean(v)) # mean
        results[k].append(np.std(v)) # var
    with open(os.path.join(results_dir, 'summary.json'), 'w+') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()



def special_args(args):
    if args.prompt_algo == 'CoCoOp':
        args.batch_size = 1
        args.eval_scaler = 1
    if args.prompt_algo == 'OTP':
        args.fed_algo = 'FedOTP'
        args.num_prompt = 2
    if args.prompt_algo == 'BPL':
        args.batch_size = 1
        args.eval_scaler = 1
    if args.prompt_algo == 'ProDA':
        args.prompt_batch_size = 1
    if args.prompt_algo == 'CLIP':
        args.ctx_init = 'a photo of a'
    if args.central == 'true':
        args.fed_algo = 'Central'

    return args

def main(args):
    if args.deterministic == 'true':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    if args.verbose:
        log.level = 'verbose'
    elif args.verbose2:
        log.level = 'debug'
    for i in range(args.times):
        server = get_server(args, i)
        start = time.time()
        server.run()
        end = time.time()
        log.info(f'Experiment run: {i}, Total time ellapsed: {int(end-start)}s')
    parse_results(args)



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-task', "--task", type=str, default='class',
                                        choices=['class', 'seg', ], )
    parser.add_argument('-bench', "--benchmark", type=str, default="global",
                        choices=['dual', 'global', 'personal',
                                 'base2novel', 'xdomain', 'multidomain', 'xdataset'])
    parser.add_argument('-falg', "--fed_algo", type=str, default='FedAvg',
                                        choices=['FedAvg', 'FedOTP', ], )
    parser.add_argument('-palg', "--prompt_algo", type=str, default="CoOp",
                                        choices=['CLIP', 'CoOp', 'CoCoOp', 'PLOT',
                                                 'ALIGN', 'ProDA', 'ProGrad', 'PromptSRC',
                                                 'BPL', 'KgCoOp','OTP',
                                                 'DenseCoOp'])
    parser.add_argument('-ctr', "--central", type=str, default='false')
    parser.add_argument('-did', "--device_id", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='summary')
    parser.add_argument('-slurm', "--slurm", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-detm', "--deterministic", type=str, default='true', choices=['true', 'false'])
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--verbose2", action='store_true')
    # model
    parser.add_argument('-ibb', "--image_backbone", type=str, default='RN50',
                                                    choices=['RN50', 'RN101', 'ViT-B/16'])
    parser.add_argument('-m', "--model", type=str, default="resnet50")
    # dataset
    parser.add_argument('-data', "--dataset", type=str, default="caltech101")
    parser.add_argument('-tdata', "--target_dataset", type=str, default="caltech101")
    parser.add_argument('-root', "--data_root", type=str, default="~/data/prompt")
    parser.add_argument('-dnt', "--num_shot", type=int, default=1)
    parser.add_argument('-dns', "--num_shards", type=int, default=10)
    parser.add_argument('-dsm', "--split_mode", type=str, default='dirichlet',
                        choices=['dirichlet', 'iid', 'task','predefined'])
    parser.add_argument('-dsa', "--split_alpha", type=float, default=1)
    parser.add_argument('-dsb', "--split_beta", type=float, default=1)
    parser.add_argument('-dtf', "--data_transform", type=str, default="default", choices=['default', 'randaug'])
    parser.add_argument('-ddl', "--drop_last", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-dpl', "--parallel", type=str, default='true', choices=['true', 'false'])
    parser.add_argument('-dnw', "--num_workers", type=int, default=8)
    # general federated learning settings
    parser.add_argument('-nc', "--num_clients", type=int, default=4)
    parser.add_argument('-cevl', "--client_eval", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-le', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-tf', "--train_fraction", type=float, default=1.)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0)
    # optimization
    parser.add_argument('-opn', "--optim_name", type=str, default='sgd')
    parser.add_argument('-orho', "--optim_rho", type=float, default=0.01)
    parser.add_argument('-lrs', "--lr_scheduler", type=str, default='', choices=['', 'cos'])
    parser.add_argument('-lbs', "--batch_size", type=int, default=8)
    parser.add_argument('-lvbs', "--eval_scaler", type=int, default=1)
    parser.add_argument('-evrds', "--eval_rounds", type=int, default=1)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-omt', "--optim_momentum", type=float, default=0.9,)
    parser.add_argument('-owd', "--optim_weight_decay", type=float, default=0.0001,)
    parser.add_argument('-lrd', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-lrdg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-prec', "--precision", type=str, default="fp32", choices=['fp32', 'fp16', 'amp'])
    parser.add_argument('-iws', "--init_weights", type=str, default=None,)
    parser.add_argument('-lst', "--loss_type", type=str, default='ce', choices=['ce', 'bce'])
    parser.add_argument('-gcn', "--grad_clipping_norm", type=float, default=0.,)
    parser.add_argument('-seed', "--seed", type=int, default=0)
    # prompt learning general settings
    parser.add_argument('-npt', "--num_prompt", type=int, default=1)
    parser.add_argument('-nptv', "--num_prompt_vision", type=int, default=1)
    parser.add_argument('-nctx', "--num_context", type=int, default=4)
    parser.add_argument('-ctp', "--class_token_position", type=str, default='end')
    parser.add_argument('-csc', "--class_specific_context", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-cti', "--ctx_init", type=str, default='')
    # prompt learning algorithms' settings
    parser.add_argument('-pbsz', "--prompt_batch_size", type=int, default=0) # ProDA
    # segmentation task settings
    parser.add_argument('-stls', "--seg_text_loss_scale", type=float, default=1) #
    # FL algorithms' settings

    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    if args.slurm == 'true':
        print(f'Job {args.prompt_algo} allocated GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        args.device_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    args = special_args(args)
    main(args)