import torch
import torch.nn as nn
import math
import numpy as np
import copy

from torch.nn import functional as F
from matplotlib import pyplot as plt


class DivergeError(ValueError):
    """ Training loss diverged to NaN.  """

def topk(output, target, k=(1, ), count=False):
    if output.dim() == 3:
        output = output.mean(dim=0)
    _, pred = output.topk(max(k), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    batch = 1 if count else target.size(0)
    return [float(correct[:k].sum()) / batch for i, k in enumerate(k)]

def entropy(output, target, ntokens):
    return [torch.nn.CrossEntropyLoss()(output.view(-1, ntokens), target)]

def unit(value, asint=False, base=1024):
    if value == 0:
        return value
    exp = math.floor(math.log(abs(value), base))
    value = value / 1024 ** exp
    value = int(value) if asint else f'{value:.2f}'
    return f'{value}{" KMGTP"[exp]}'.replace(' ', '')


class AccuracyCounter:
    supported_tasks = ['image', 'language']

    def __init__(self, num, k=(1, ), task='image', ntokens=None, num_classes=10):
        super().__init__()
        self.num = num
        self.k = k
        self.correct = [0] * len(k)
        self.entropies = []
        self.size = 0
        if task not in self.supported_tasks:
            raise ValueError(
                f'Task {task!r} not in supprted list {self.supported_tasks}.')
        self.task = task
        self._ntokens = ntokens
        self.num_classes = num_classes
        self.class_accs = {c:0 for c in range(self.num_classes)}

    def add(self, output, target):
        self.size += target.size(0)
        if output.dim == 3:
            output = output.mean(dim=1)
        if self.task == 'image':
            for i, a in enumerate(topk(output, target, self.k, True)):
                self.correct[i] += a
            #self.per_class_accs(output, target)
        if self.task == 'language':
            self.entropies.append(entropy(output, target, self._ntokens))

    def logout(self):
        if self.task == 'image':
            return self.accuracies()
        if self.task == 'language':
            return self.entropy()
        raise ValueError

    def entropy(self):
        return np.mean(self.entropies)

    def accuracies(self):
        for i in range(len(self.k)):
            yield self.correct[i] / self.size

    def errors(self):
        for a in self.accuracies():
            yield 1 - a

    def progress(self):
        return self.size / self.num

    def per_class_accs(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        for c in range(self.num_classes):
            correct = ((labels==preds)*(labels==c)).sum()
            self.class_accs[c] += correct

class MovingAverage:
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.items = []

    def add(self, value):
        self.items.append(float(value))
        if len(self.items) > self.num:
            self.items = self.items[-self.num:]

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

    def flush(self):
        self.items = []

    def __format__(self, mode):
        text = f'{self.mean():.5f}'
        if 's' not in mode:
            return text
        return text + f'±{self.std() * 100:.2f}%'

    def __float__(self):
        return self.mean()



def batch_loss(logits, labels, reduction):
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0) # BPL returns loigt.dim = 3 #[L, bs, dim]
    loss = 0.
    for batch_logits in logits:
        loss += criterion(batch_logits, labels)
    return loss



def build_loss_fn(base_probs, loss_type='ce', tau=1.0, reduction='mean'):
    """Builds the loss function.
    Args:
        base_probs: Base probabilities to use in the logit-adjusted loss.
        tau: Temperature scaling parameter for the base probabilities.
        loss_type: the loss type for training. options:['lc', 'ce', 'bce']
    Returns:
        A loss function with signature loss(labels, logits).
    """
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    def lc_loss_fn(logits, labels):
        """ logit calibration loss
        """
        base_probs[base_probs==0] = 1 # avoid deviding by zero
        logits = logits - tau * torch.pow(base_probs, -1/4)
        loss = batch_loss(logits, labels, reduction)
        return loss
    def bce_loss_fn(logits, labels):
        """ balanced cross entropy loss
        """
        logits = logits + tau * torch.log(base_probs + 1e-12) # avoid underflow
        loss = batch_loss(logits, labels, reduction)
        return loss
    def ce_loss_fn(logits, labels):
        """ cross entropy loss
        """
        loss = batch_loss(logits, labels, reduction)
        return loss
    def pc_loss_fn(logits, labels):
        """ partial class loss: zero-out logits  from missing classes
            but leave imbalanced classes untouched
        """
        class_filter = (base_probs>=1e-5).int()
        logits = logits * class_filter
        loss = batch_loss(logits, labels, reduction)
        return loss
    loss_maps = {
        'lc': lc_loss_fn,
        'ce': ce_loss_fn,
        'bce': bce_loss_fn,
        'pc': pc_loss_fn,
    }
    return loss_maps[loss_type]


def eval_global(model, dataloader, device, precision='amp'):
    model.eval()
    model.set_classifier()
    ac = AccuracyCounter(
        len(dataloader.dataset), (1, 5),
        task='image',
        num_classes=len(dataloader.dataset.dataset.classes))
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            if precision=='amp':
                with torch.cuda.amp.autocast():
                    output = model(images, labels, test=True)
            else:
                output = model(images, labels, test=True)
            ac.add(output, labels)

    return ac.logout()

def eval_base_novel(model, dataloader, device, train_classnames, test_classnames, precision='fp32'):
    """ evaluation on base or novel classes for benchmarking 'base2novel' task
    """
    # reinit embedding of prompt learner with test classnames
    model.prompt_learner.init_embedding(test_classnames)
    accs = eval_global(model, dataloader, device, precision)
    # revert to train classnames
    model.prompt_learner.init_embedding(train_classnames)
    return accs


def eval_personal(model, states, weights, dataloaders, device, precision='fp32'):
    """ evaluation personalized accuracy
    """
    client_accs = {}
    for c, state in states.items():
        model.prompt_learner.load_state_dict(state, strict=False)
        dataloader = dataloaders[c]
        top1,_ = eval_global(model, dataloader, device, precision)
        client_accs[c] = top1
    # weighted average accuracy
    avg_acc = np.sum([client_accs[c] * w  for c,w in weights.items()])
    return avg_acc, client_accs

def eval_domains(model, dataloaders, device, precision='fp32'):
    """ evaluation multi-domain accuracy
    """
    domain_accs, weights = {}, {}
    for name, dataloader in dataloaders.items():
        top1,_ = eval_global(model, dataloader, device, precision)
        domain_accs[name] = top1
        weights[name] = len(dataloader.dataset.targets)
    # weighted average accuracy
    total_samples = sum(weights.values())
    weights = {k:v / total_samples for k,v in weights.items()}
    avg_acc = np.sum([domain_accs[name] * w  for name,w in weights.items()])
    return avg_acc, domain_accs

class JSDiv(nn.Module):

    def __init__(self):
        super(JSDiv, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=0)
        net_2_probs=  F.softmax(net_2_logits, dim=0)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean")

        return (0.5 * loss)


def calculate_js_loss(logit_context, logit_target, mean_context, sigma_context, mean_target, sigma_target):
    log_var_context = 2 * (sigma_context.log())
    log_var_target = 2 * (sigma_target.log())

    context_B = logit_context.size(1)
    target_B = logit_target.size(1)
    logit_target_pred = F.softmax(logit_target, dim = -1)
    logit_context_pred = F.softmax(logit_context, dim = -1)

    uncertainty_context_avg = (-1.0 * torch.sum(logit_context_pred.mean(0) * logit_context_pred.mean(0).log())/context_B).detach()
    uncertainty_target_avg = (-1.0 * torch.sum(logit_target_pred.mean(0) * logit_target_pred.mean(0).log())/target_B).detach()

    alpha = uncertainty_context_avg / (uncertainty_context_avg + uncertainty_target_avg)
    alpha_var = ((1 - alpha) * (-1 * log_var_context).exp() + alpha * (-1 * log_var_target).exp())**(-1)
    alpha_mean = alpha_var * ((1 - alpha) * (-1 * log_var_context).exp() * mean_context + alpha * (-1 * log_var_target).exp() * mean_target)


    skew_uncertain_loss = torch.sum((((1 - alpha) * log_var_context.exp() + \
                                      alpha * log_var_target.exp()) * (alpha_var ** (-1)) + \
                                     (alpha_var.log() - (1-alpha) * log_var_context - alpha  * log_var_target) + \
                                    (1-alpha) * ((alpha_mean - mean_context)**2) * (alpha_var**(-1)) + \
                                    alpha * ((alpha_mean - mean_target)**2) * (alpha_var**(-1)) - 1) * 0.5 )



    return skew_uncertain_loss


def reg_loss(text_features, text_fixed_ebds):
    return F.l1_loss(text_features, text_fixed_ebds, reduction='mean')


# Use PCA to reduce dimensionality

def svd(X, n_components=2):
    # using SVD to compute eigenvectors and eigenvalues
    # M = np.mean(X, axis=0)
    # X = X - M
    # U, S, Vt = np.linalg.svd(X)
    U, S, Vt = torch.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]


def plot_id_ood_gap(all_img_features, all_text_features):
    features_2d = svd(np.concatenate([all_img_features, all_text_features], 0))
    num_feat = len(all_img_features)
    plt.figure(figsize=(5, 5))
    plt.scatter(features_2d[:-num_feat, 0], features_2d[:-num_feat, 1], c='red')
    plt.scatter(features_2d[-num_feat:, 0], features_2d[-num_feat:, 1], c='blue')
    # connect the dots
    for i in range(num_feat):
        plt.plot([features_2d[i, 0], features_2d[num_feat+i, 0]],
                 [features_2d[i, 1], features_2d[num_feat+i, 1]],
                 c='black', alpha=0.1)

state_dict_filters = {
    'CoOp': ['ctx'],
    'CoCoOp': ['ctx', 'meta_net.linear1.weight','meta_net.linear1.bias',
               'meta_net.linear2.weight','meta_net.linear2.bias',],
    'PromptSRC': ['ctx'],
    'PLOT': ['ctx'],
    'ProDA': ['ctx'],
    'ProGrad': ['ctx'],
    'OTP': ['ctx'],
    'KgCoOp': ['ctx', ],
    'BPL': ['ctx', ], # TODO
}

def filter_states(algo, state):
    keys = state_dict_filters[algo]
    if state in keys:
        return True
    else:
        return False
