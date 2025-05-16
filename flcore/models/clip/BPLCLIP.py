import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BPLPromptLearner

class BPLCLIP(BaseCLIP):
    """ Bayesian Prompt Learning for Image-Language Model Generalization
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = BPLPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.L = self.prompt_learner.L

    def forward(self, image, labels=None, test=False):
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = torch.tile(tokenized_prompts, (self.L, 1))
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))  # 1 x 512
        image_features = image_features[0] # only use the pooled features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, mu, logvar = self.prompt_learner(image_features)  # L x NumClass x Length x DIM
        _, NumClass, Length, dim = prompts.shape
        prompts = prompts.view(-1, Length, dim)  # (L * NumClass) x Length x DIM
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features.unsqueeze(0).expand((self.L, -1, -1))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(
            -1, NumClass, text_features.shape[-1]
        )  # L * NumClass x DIM

        logits = logit_scale * torch.einsum("LBD,LCD->LBC", image_features, text_features)

        log_p_y = torch.log_softmax(logits, dim=-1)

        if not test:
            # self.kl_loss = 0.001 * self.kl_divergence(mu, logvar).mean(0)
            self.kl_loss = self.kl_divergence(mu, logvar).mean(0)
            return logits
        else:
            average_prediction = torch.logsumexp(log_p_y, dim=0) - torch.log(
                torch.Tensor([self.L]).type_as(logits)
            )
            return average_prediction

    def kl_divergence(self, mu, logvar):
        prior_mu = torch.zeros_like(mu)
        prior_std = torch.ones_like(logvar)

        prior = torch.distributions.Normal(loc=prior_mu, scale=prior_std)
        post = torch.distributions.Normal(loc=mu, scale=logvar.exp().sqrt())

        dist = torch.distributions.kl_divergence(post, prior).mean(dim=-1)
        return dist

    def nll(self, logits, targets):
        task_log_py = (logits * targets).sum(dim=-1)
        return task_log_py

    def add_loss(self,):
        return self.kl_loss