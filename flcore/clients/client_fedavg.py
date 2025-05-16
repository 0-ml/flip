import copy
import torch
import numpy as np
import time
import collections
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from .client_base import Client
from ..utils import (MovingAverage, AccuracyCounter, DivergeError,
                          topk, IoU, unit, build_loss_fn, eval_global, eval_base_novel,
                          collect_state)
from ..pretty.logger import log

class ClientFedAvg(Client):
    len_history = 100
    def __init__(self, args, id, trainloader, testloader, **kwargs):
        super().__init__(args, id, trainloader, testloader, **kwargs)

    def local_train(self,):
        begin_time = time.time()
        msg = f'c: {self.client_id}'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        avg_add_losses = MovingAverage(self.len_history)
        self.optimizer.state = collections.defaultdict(dict)
        self.optimizer.zero_grad()
        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = self.rounds - 1
            self.optimizer.zero_grad()
            if self.optim_name != 'sam':
                self.optimizer.step()  # disable warning on the next line...
            self.lr_scheduler.step()
        self.init_state = copy.deepcopy(self.model.prompt_learner).state_dict()
        flops, steps = 0, 0
        result = {
            'flops.model': flops,
            'flops.total': flops * steps * self.batch_size,
        }
        init_state = {
            k: v.to(self.device, copy=True) for k, v in self.init_state.items()}
        if self.task == 'seg':
            self.model.decoder.train()
        self.model.update_global_text_feats()
        try:
            for e in range(self.local_epochs):
                for data, target in self.trainloader:
                    self._step(data, target, init_state, avg_losses, avg_add_losses, avg_accs, self.rounds)
                if self.central:
                    self.round_eval(avg_losses, avg_add_losses, e)
        except DivergeError as e:
            log.verbose(f'{msg}, diverged to NaN.')
            return {'status': 'error', 'exception': e, **result}

        result.update({
            'status': 'normal',
            'state': collect_state(self.model, self.task, self.prompt_algo),
            'accuracy': float(avg_accs.mean()),
            'loss': float(avg_losses.mean()),
            'num_train_samples': len(self.trainloader.dataset.targets)
        })
        if self.client_eval:
            if self.bench == 'base2novel':
                top1_base, _ = eval_base_novel(self.model,  self.testloader_base, self.device,
                                  self.train_classnames, self.test_classnames_base, self.precision)
                result.update({'eval_acc':top1_base,})

                top1_novel, _ = eval_base_novel(self.model,  self.testloader_novel, self.device,
                                  self.train_classnames, self.test_classnames_novel, self.precision)
                msg += f', eval_novel: {top1_novel:.2%}'
                result.update({'eval_novel':top1_novel})
            else:
                top1, _ = eval_global(self.model,  self.testloader, self.device,
                                      self.precision, self.task)
                result.update({
                    'eval_acc':top1,
                })
                msg += f', eval: {top1:.2%}'
        end_time = time.time()
        log.info(
            f'{msg}, train: {float(avg_accs.mean()):.2%}, '
            f'ep: {self.local_epochs}, '
            f'lr: {self._get_lr():.4f}, flops: {unit(flops)}, '
            f'loss: {(avg_losses.mean()):.4f}, '
            f'add_loss: {(avg_add_losses.mean()):.6f}, '
            f'time:{end_time - begin_time:.2f}s.')

        self.round_epilog()
        return result

    def _step(self, data, target, init_state, avg_losses, avg_add_losses, avg_accs, rounds):
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
        else:
            if self.optim_name == 'sam':
                output = self.model(data, target, test=False)
                loss = self.loss_func(output, target)
                add_loss = self.model.add_loss()
                loss += add_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                output = self.model(data, target, test=False)
                loss = self.loss_func(output, target)
                loss.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                output = self.model(data, target, test=False)
                loss = self.loss_func(output, target)
                add_loss = self.model.add_loss()
                loss += add_loss

                if torch.isnan(loss).any():
                    raise DivergeError('Training loss diverged to NaN.')

                loss.backward()
                self.grad_func(init_state)
                if self.grad_clipping_norm > 0:
                    self.grad_clip()
                self.optimizer.step()
        train_acc = self.metric(output, target)
        avg_losses.add(loss)
        avg_add_losses.add(add_loss)
        avg_accs.add(train_acc)

    def round_epilog(self,):
        if not self.central:
            self.keep_states()
            del self.model
            del self.optimizer
            del self.lr_scheduler
            del self.init_state

    def round_eval(self, avg_losses, avg_add_losses, cur_epoch):
        top1, _ = eval_global(self.model,  self.testloader, self.device,
                              self.precision, self.task)
        msg = f'{self.prompt_algo}, c: {self.client_id}'
        log.verbose(
            f'{msg}, eval: {float(top1):.2%}, '
            f'total_epochs: {self.local_epochs}, '
            f'cur_epoch: {cur_epoch}, '
            f'lr: {self._get_lr():.4f}, '
            f'loss: {(avg_losses.mean()):.4f}, '
            f'add_loss: {(avg_add_losses.mean()):.4f}')
        if self.lr_scheduler:
            self.lr_scheduler.step()
