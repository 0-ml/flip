import torch
import torch.nn as nn
import os
import numpy as np
import h5py
import copy
import time
import random
import datetime
import json
from torch.cuda.amp import GradScaler, autocast

from ..models.utils import load_pretrained_weights
from ..models.clip import clip_maps
from ..models.clip.clip import load_clip_to_cpu
from ..datasets import datasets_map
from ..datasets.info import INFO
from ..utils import unit, AccuracyCounter, eval_global, eval_base_novel
from ..pretty.history import History
from ..pretty.logger import log

class ServerBase(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.task = args.task
        self.args = args
        self.bench = args.benchmark
        self.fed_algo = args.fed_algo
        self.prompt_algo = args.prompt_algo
        self.central = args.central == 'true'
        self.device = torch.device('cuda', args.device_id)
        self.state_device = torch.device('cpu')
        self.dataset = args.dataset
        self.target_dataset = args.target_dataset
        self.img_folder = INFO[args.dataset]['img_folder']
        self.image_size = INFO[args.dataset]['shape'][-1]
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = 1 if self.central else args.num_clients
        self.num_shards = args.num_shards
        self.split_mode = args.split_mode
        self.data_root = args.data_root
        self.data_transform = args.data_transform
        self.drop_last = args.drop_last == 'true'
        self.split_alpha = args.split_alpha
        self.split_beta = args.split_beta
        self.train_fraction = args.train_fraction
        self.num_join_clients = int(self.num_clients * self.train_fraction)
        self.current_num_join_clients = self.num_join_clients
        self.grad_clipping_norm = args.grad_clipping_norm
        self.seed = times
        self.clients = []
        self.num_shot = args.num_shot
        self.num_workers = args.num_workers
        self.client_drop_rate = args.client_drop_rate
        self.exp_name = os.path.join(
            self.dataset, args.image_backbone, self.fed_algo, self.prompt_algo)
        dtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.tbname = os.path.join(self.exp_name, dtime)
        self.tb = History(self.tbname)
        self.client_eval = args.client_eval == 'true'
        self.precision = args.precision
        self.eval_scaler = args.eval_scaler
        self.eval_rounds = args.eval_rounds
        self.optim_name = args.optim_name
        self.momentum = args.optim_momentum
        self.weight_decay = args.optim_weight_decay
        self.learning_rate_scheduler = args.lr_scheduler
        self.loss_type = self.args.loss_type

    def init_clients(self, clientObj):
        self.init_dataset()
        self.init_model()
        for i in range(self.num_clients):
            client = clientObj(self.args,
                            id=i,
                            trainloader=self.trainloaders[i],
                            testloader=self.testloader)
            self.set_client_attr(client)
            self.clients.append(client)

    def set_client_attr(self, client):
        client.train_classnames = self.train_classnames
        client.test_classnames = self.test_classnames
        if self.bench == 'base2novel':
            client.testloader_base = self.testloader_base
            client.testloader_novel = self.testloader_novel
            client.test_classnames_base = self.test_classnames_base
            client.test_classnames_novel = self.test_classnames_novel


    def init_model(self,):
        log.debug(f"Loading CLIP (backbone: {self.args.image_backbone})")
        clip_model = load_clip_to_cpu(self.args.image_backbone)

        if self.args.precision == "fp32" or self.args.precision == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False)

        log.debug("Building custom CLIP")

        self.model = clip_maps[self.prompt_algo](self.args, self.train_classnames,
                                                 clip_model)

        print("Turning off gradients in both the image and the text encoder")
        def requires_grad_filter(name):
            requires_list = ['prompt_learner', 'decoder',]
            excludes_list = ['embedding_func', 'ZS_image_encoder']
            for k in requires_list:
                if k in name:
                    for e in excludes_list:
                        if e in name:
                            return False
                    return True
            return False
        for name, param in self.model.named_parameters():
            param.requires_grad_(requires_grad_filter(name))
            print(name, param.requires_grad)

        if self.args.init_weights:
            load_pretrained_weights(self.model.prompt_learner, self.args.init_weights)

        if self.args.dataset == "ImageNet":
            cuda_device = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(cuda_device)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)

        self.scaler = GradScaler() if self.args.precision == "amp" else None

    def init_dataset(self,):
        self.dataset_params = { 'bench': self.bench,
            'name': self.dataset, 'split': 'train', 'batch_size': self.batch_size,
            'num_clients': self.num_clients, 'num_shards': self.num_shards,
            'split_mode': self.split_mode, 'alpha': self.split_alpha,
            'beta': self.split_beta, 'parallel': True, 'data_dir': self.data_root,
            'num_shot': self.num_shot, 'data_transform': self.data_transform,
            'img_folder': self.img_folder, 'drop_last': self.drop_last,
            'seed': self.seed, 'num_workers': self.num_workers, 'subsample': 'base',
        }
        Dataset = datasets_map[self.dataset]
        self.trainloaders = Dataset(**self.dataset_params)
        self.test_params = copy.deepcopy(self.dataset_params)
        self.val_params = copy.deepcopy(self.dataset_params)
        self.test_params['split'] = 'test'
        self.test_params['drop_last'] = False
        self.test_params['subsample'] = None
        self.test_params['batch_size'] = self.batch_size * self.eval_scaler
        # if self.bench == 'xdomain':
        #     self.test_params['name'] = self.target_dataset
        #     self.val_params['name'] = self.target_dataset
        self.testloader = Dataset(**self.test_params) # test all classes for eval_global
        self.val_params['split'] = 'val'
        self.val_params['drop_last'] = False
        self.val_params['batch_size'] = self.batch_size * self.eval_scaler
        self.valloader = Dataset(**self.val_params)
        if self.bench == 'dual' or self.bench == 'personal':
            self.test_params_per = copy.deepcopy(self.dataset_params)
            self.test_params_per['split'] = 'personal'
            self.test_params_per['drop_last'] = False
            self.test_params_per['subsample'] = None
            self.test_params_per['batch_size'] = self.batch_size * self.eval_scaler
            self.testloaders_per = Dataset(**self.test_params_per)
        if self.bench == 'multidomain':
            self.testloaders_mdomain = self.testloader
            self.testloader = self.testloader['real']

        self.train_classnames = list(self.trainloaders[0].dataset.dataset.classes)
        self.test_classnames = list(self.testloader.dataset.dataset.classes)

        if self.bench == 'base2novel':
            self.test_params['subsample'] = 'base'
            self.testloader_base = Dataset(**self.test_params)
            self.test_params['subsample'] = 'new'
            self.testloader_novel = Dataset(**self.test_params)
            self.test_classnames_base = list(self.testloader_base.dataset.dataset.classes)
            self.test_classnames_novel = list(self.testloader_novel.dataset.dataset.classes)


    def select_clients(self):
        self.cur_join_clients = self.num_join_clients
        selected_client_ids = list(np.random.choice(list(range(self.num_clients)),
                                                self.cur_join_clients, replace=False))
        return selected_client_ids

    def eval_personal(self,):
        pass

    def eval_multiple(self, *args, **kwargs):
        pass

    def receive_results(self, results):
        assert (len(self.selected_client_ids) > 0)

        active_client_ids = random.sample(self.selected_client_ids,
                        int((1-self.client_drop_rate) * self.cur_join_clients))
        states, weights, losses, accs = {}, {}, [], []
        tot_samples = 0
        for c in active_client_ids:
            losses.append(results[c]['loss'])
            accs.append(results[c]['accuracy'])
            tot_samples += results[c]['num_train_samples']
            weights[c] = results[c]['num_train_samples']
            states[c] = {
                k: v.to(self.state_device) for k, v in results[c]['state'].items()}
        for i, w in weights.items():
            weights[i] = w / tot_samples
        avg_loss, avg_acc = np.mean(losses), np.mean(accs)
        self.tb.add_scalar('train/client/avg_loss', avg_loss, self.rounds)
        self.tb.add_scalar('train/client/avg_acc', avg_acc, self.rounds)
        if self.client_eval:
            self.tb_client_accs(results, self.rounds)
        info = (
            f'round {self.rounds}, train loss: '
            f'{avg_loss:.3f}±{np.std(losses) / avg_loss:.1%}'
            )
        return states, weights, info

    def aggregate(self, states, weights):
        assert (len(states) > 0)
        avg_state = {}
        keys = list(states[list(states)[0]])
        for k in keys:
            s = [s[k].to(self.state_device) * weights[c] for c, s in states.items()]
            avg_state[k] = sum(s).to(self.state_device)
        return avg_state

    def save_checkpoint(self, info, name):
        torch.save(info, name)

    def grad_func(self,):
        pass

    def run(self,):
        raise NotImplementedError

    def tb_client_accs(self, results, rounds):
        for c, rv in results.items():
            self.tb.add_scalar(f'eval/client_{c}_acc', rv['eval_acc'], rounds)


    def summarize(self,):
        if self.bench == 'base2novel':
            exp_results = {'base': self.best_acc_base,
                           'novel':self.best_acc_novel,
                           'hm':self.best_acc}
        elif self.bench == 'dual':
            exp_results = {'global': self.best_acc,
                           'personal': self.best_acc_per,
                           'hm': self.best_acc_hm}
        elif self.bench == 'xdomain' or self.bench == 'global':
            exp_results = {'global': self.best_acc}
        elif self.bench == 'personal':
            exp_results = {'personal': float(self.best_acc_per)}
        elif self.bench == 'multidomain':
            exp_results = {'multidomain': float(self.best_acc),
                           **self.best_acc_mds}
        else:
            raise  NotImplementedError

        print(exp_results)
        results_dir = os.path.join('results', self.exp_name, f'shot_{self.num_shot}')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'exps_seed_{self.seed}.json')
        with open(results_path, 'w+', encoding='utf-8') as f:
            json.dump(exp_results, f, ensure_ascii=False, indent=4)
        f.close()
