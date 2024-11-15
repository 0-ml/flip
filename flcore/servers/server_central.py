import time
import torch
import numpy as np
import torch.nn as nn
import copy
from statistics import harmonic_mean as hmean
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from ..clients.client_fedavg import ClientFedAvg
from ..servers.server_base import ServerBase
from ..utils import DivergeError, eval_global, eval_base_novel, eval_personal, svd, eval_domains
from ..pretty.logger import log
from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
from ..utils import (MovingAverage, unit, topk, build_loss_fn, Metric)
from ..datasets.info import INFO

class Central(ServerBase):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.init_clients(ClientFedAvg)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.
        self.best_acc_novel = 0.
        self.best_acc_hm = 0.
        self.best_acc_per = 0.
        self.best_acc_mds = {}
        self.id_text_feats = {}
        self.ood_text_feats = {}
        self.vis_plot = False
        self.trainloader = self.trainloaders[0] # central mode has only 1 trainloader
        self.len_history = 100

    def run(self):
        log.info('Starting Centralized Training...')
        self.central_init()
        for i in range(self.global_rounds+1):
            self.cur_epoch = i
            start_time = time.time()
            _ = self.train_epoch()
            top1, _ = eval_global(self.model, self.valloader, self.device,
                                  self.precision, self.task)
            self.best_acc = top1 if top1 > self.best_acc else self.best_acc
            self.tb.add_scalar('eval/top1', top1, self.cur_epoch)
            self.tb.add_scalar('eval/best', self.best_acc, self.cur_epoch)
            log.info(f'epoch: {i}, top1: {top1:.2%}, '
                     f'elapsed: {time.time() - start_time:.1f}s, '
                     f'best_acc: {self.best_acc:.2%}.')
        self.summarize()

    def central_init(self,):
        self._init_opt(self.model, self.optim_name, self.learning_rate)
        self._init_metric()
        self._init_loss()

    def _init_loss(self, ):
        if self.task == 'class':
            self.loss_func = self.adjusted_loss(self.loss_type, 'mean')
        elif self.task == 'seg':
            ignore_index = INFO[self.dataset]['ignore_index']
            self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)


    def _init_opt(self, model, optim, lr):
        params = filter(lambda p: p.requires_grad, model.parameters())
        if optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        else:
            raise 'Unknown optimizer!'
        if self.learning_rate_scheduler == 'cos':
            self.lr_scheduler = CosineAnnealingLR(self.optimizer,
                                                    T_max = self.args.global_rounds)
        else:
            self.lr_scheduler = None

    def train_epoch(self, ):
        begin_time = time.time()
        msg = f'central'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        avg_add_losses = MovingAverage(self.len_history)
        self.optimizer.zero_grad()
        if self.lr_scheduler:
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.lr_scheduler.step()
        flops, steps = 0, 0
        result = {
            'flops.model': flops,
            'flops.total': flops * steps * self.batch_size,
        }
        try:
            for data, target in self.trainloader:
                self._step(data, target, avg_losses, avg_add_losses, avg_accs)
        except DivergeError as e:
            log.verbose(f'{msg}, diverged to NaN.')
            return {'status': 'error', 'exception': e, **result}

        result.update({
            'state': {
                k: v.detach().clone().cpu()
                for k, v in self.model.prompt_learner.state_dict().items()},
            'accuracy': float(avg_accs.mean()),
            'loss': float(avg_losses.mean()),
            'num_train_samples': len(self.trainloader.dataset.targets)
        })
        end_time = time.time()
        log.info(
            f'{msg}, train: {float(avg_accs.mean()):.2%}, '
            f'ep: {self.cur_epoch}, '
            f'lr: {self._get_lr():.4f}, flops: {unit(flops)}, '
            f'loss: {(avg_losses.mean()):.4f}, '
            f'add_loss: {(avg_add_losses.mean()):.6f}, '
            f'time:{end_time - begin_time:.2f}s.')

        return result

    def _step(self, data, target, avg_losses, avg_add_losses, avg_accs):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        if self.args.precision == "amp":
            with autocast():
                output = self.model(data, target, test=False)
                loss = self.loss_func(output, target)
                add_loss = self.model.add_loss()
                loss += add_loss
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        train_acc = self.metric(output, target)
        avg_losses.add(loss)
        avg_add_losses.add(add_loss)
        avg_accs.add(train_acc)

    def adjusted_loss(self, loss_type='bce', reduction='mean'):
        stats = torch.tensor(self.trainloader.stats)
        base_probs = (stats/stats.sum()).to(self.device)
        tau = 1
        return build_loss_fn(base_probs, loss_type=loss_type, tau=tau,
                                                    reduction=reduction)

    def _get_lr(self):
        if self.lr_scheduler:
            return self.lr_scheduler.get_last_lr()[0]
        else:
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def _init_metric(self, ):
        if self.task == 'class':
            self.metric = Metric(self.task, k=(1,), count=False)
        elif self.task == 'seg':
            num_classes = INFO[self.dataset]['num_classes']
            ignore_index = INFO[self.dataset]['ignore_index']
            self.metric = Metric(self.task, num_classes=num_classes,
                                    ignore_index=ignore_index)