import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import KgCoOpPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class KgCoOpCLIP(BaseCLIP):
    """ Visual-Language Prompt Tuning with Knowledge-guided Context Optimization
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = KgCoOpPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.meta_net = self.prompt_learner.meta_net
        self.adapter = Adapter(512, 4).to(clip_model.dtype)
        self.w = 1.

    def forward(self, image, labels=None, test=False):
        prompts = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[0]

        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_old = self.ori_embedding

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = cos(text_features,text_features_old)
        self.score = (1.0-torch.mean(score)) * self.w

        return logits

    def add_loss(self, ):
        return self.score