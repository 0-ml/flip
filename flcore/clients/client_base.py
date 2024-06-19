import copy
import torch
import torch.nn as nn
import numpy as np
import os
import time

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from sklearn import metrics
from ..utils import AccuracyCounter, build_loss_fn
from ..optimizers.sam import SAM

class Client(object):
    """
    Base class for FL clients.
    """
    def __init__(self, args, id, trainloader, testloader, **kwargs):
        self.args = args
        self.bench = args.benchmark
        self.fed_algo = args.fed_algo
        self.prompt_algo = args.prompt_algo
        self.dataset = args.dataset
        self.device = torch.device('cuda', args.device_id)
        self.client_id = id
        self.save_folder_name = args.save_folder_name
        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.learning_rate_scheduler = args.lr_scheduler
        self.local_epochs = args.local_epochs
        self.rounds = 0
        self.optim_name = args.optim_name
        self.optim_rho = args.optim_rho
        self.momentum = args.optim_momentum
        self.weight_decay = args.optim_weight_decay
        self.grad_clipping_norm = args.grad_clipping_norm
        self.task = 'image'
        self.client_eval = args.client_eval == 'true'
        self.central = args.central == 'true'
        self.loss_type = self.args.loss_type
        self.precision = args.precision
        self.scaler = GradScaler() if self.precision == "amp" else None


    def init(self, global_model, rounds):
        self.rounds = rounds
        self.model = copy.deepcopy(global_model)
        self.load_states()
        self._init_opt(self.model.prompt_learner, self.optim_name, self.learning_rate)
        self.loss_func = self.adjusted_loss(self.loss_type, 'mean')

    def _init_opt(self, model, optim, lr):
        # params = [{'params': model.parameters(), 'lr': lr}]
        # params = model.parameters()
        params = filter(lambda p: p.requires_grad, model.parameters())
        if optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        elif optim == 'sam':
            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(params=params, base_optimizer=base_optimizer,
                                 rho=self.optim_rho , lr=lr, momentum=self.momentum,
                                 weight_decay=self.weight_decay)
        else:
            raise 'Unknown optimizer!'
        if self.learning_rate_scheduler == 'cos':
            self.lr_scheduler = CosineAnnealingLR(self.optimizer,
                                                    T_max = self.args.global_rounds)
        else:
            self.lr_scheduler = None

    def local_train(self,):
        raise NotImplementedError

    def set_parameters(self, model, states):
        model.load_state_dict(states, strict=False)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def _get_lr(self):
        if self.lr_scheduler:
            return self.lr_scheduler.get_last_lr()[0]
        else:
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def grad_func(self, init_state):
        pass

    def adjusted_loss(self, loss_type='bce', reduction='mean'):
        stats = torch.tensor(self.trainloader.stats)
        base_probs = (stats/stats.sum()).to(self.device)
        tau = 1
        return build_loss_fn(base_probs, loss_type=loss_type, tau=tau,
                                                    reduction=reduction)

    def load_states(self, ):
        pass

    def keep_states(self, ):
        pass