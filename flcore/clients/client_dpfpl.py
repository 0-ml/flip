import copy
import torch
import math
from torch.cuda.amp import autocast

from .client_fedavg import ClientFedAvg
from ..utils import DivergeError

def compute_full_grad(left, right, dtype):
        left_w, left_g = left.data.type(dtype), left.grad.type(dtype) / 10.0
        right_w, right_g = right.data.type(dtype), right.grad.type(dtype) / 10.0

        left_g_right_w = torch.matmul(left_g, right_w)
        m1 = left_g_right_w + torch.matmul(left_w, right_g)
        m2 = torch.matmul(left_w, torch.matmul(left_w.T, left_g_right_w))

        return m1 + m2

class ClientDPFPL(ClientFedAvg):
    len_history = 100

    def __init__(self, args, id, trainloader, testloader, **kwargs):
        super().__init__(args, id, trainloader, testloader, **kwargs)
        max_batch = self.args.batch_size
        if args.noise > 0:
            q = 1 # random sampling
            delta = 1e-5 # delta
            steps = args.optim_rounds # number of gaussian applications
            sigma = q * math.sqrt(steps * math.log(1/delta)) / args.noise
            sensitivity = args.norm_thresh / max_batch # sensitivity
            self.std = sigma * sensitivity

    def keep_states(self, ):
        # ctx[0]: global avg ctx, ctx[1]: local ctx
        self.local_prompt_U = copy.deepcopy(self.model.prompt_learner.local_u_ctx)
        self.local_prompt_V = copy.deepcopy(self.model.prompt_learner.local_v_ctx)
        self.local_prompt = copy.deepcopy(self.model.prompt_learner.local_ctx)

    def load_states(self, ):
        self.model.prompt_learner.local_u_ctx.data = self.local_prompt_U
        self.model.prompt_learner.local_v_ctx.data = self.local_prompt_V
        self.model.prompt_learner.local_ctx.data = self.local_prompt


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
                self.add_differential_privacy()
                self.optimizer.step()
        train_acc = self.metric(output, target)
        avg_losses.add(loss)
        avg_add_losses.add(add_loss)
        avg_accs.add(train_acc)

    def add_differential_privacy(self, ):
        param_dict = dict(self.model.named_parameters())
        if self.args.noise > 0:
            grad = param_dict['prompt_learner.global_ctx'].grad.data
            norm = grad.norm(2)
            if norm > self.args.norm_thresh:
                scale = self.args.norm_thresh / norm
                scale[scale>1] = 1
                param_dict['prompt_learner.global_ctx'].grad *= scale
            elif self.args.factorization in ['dpfpl']:
                grad = param_dict['prompt_learner.local_u_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.args.norm_thresh:
                    scale = self.args.norm_thresh / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_u_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_u_ctx'].grad += noise
                grad = param_dict['prompt_learner.local_v_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.args.norm_thresh:
                    scale = self.args.norm_thresh / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_v_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_v_ctx'].grad += noise

        if self.args.factorization in ['dpfpl']:
            full_grad = compute_full_grad(param_dict['prompt_learner.local_u_ctx'], param_dict['prompt_learner.local_v_ctx'], self.dtype)
            full_grad = full_grad.type(self.dtype)
            param_dict['prompt_learner.local_ctx'].grad = full_grad