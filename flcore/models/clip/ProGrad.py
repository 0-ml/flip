import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner
from .clip import load_clip_to_cpu
from .ZSCLIP import ZSCLIP

class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return kl_loss


class ProGradCLIP(BaseCLIP):
    """ Prompt-aligned Gradient for Prompt Tuning
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.n_cls = len(classnames)
        self.prompt_learner = BasePromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_prm = args.num_prompt
        self.prograd_lambda = 1.
        self.T = 1.
        self.zs_clip = self.init_zs_clip(args, classnames)
        self.prograd_loss = ProGradLoss(T=self.T)

    def init_zs_clip(self, args, classnames):
        """ init zero shot clip model
        """
        zs_clip_model = load_clip_to_cpu(args.image_backbone)
        zs_clip_model.float()
        zs_clip = ZSCLIP(args, classnames, zs_clip_model)

        print("Turning off gradients in ZS Clip model")
        for _, param in zs_clip.named_parameters():
            param.requires_grad_(False)
        return zs_clip.to(self.device)

    def forward(self, image, labels=None, test=False):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)

        image_feature_pool = image_features[0]
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_pool @ text_features.t()
        # prograd loss
        if not test:
            zs_logits = self.zs_clip(image)
            self.pg_loss = self.prograd_loss(logits, zs_logits.detach(), labels) * self.prograd_lambda
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.n_prm, self.n_cls).mean(dim=1)

        return logits

    def add_loss(self):
        return self.pg_loss