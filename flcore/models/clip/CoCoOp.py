import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import CoCoOpPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO



class CoCoOpCLIP(BaseCLIP):
    """ Conditional Context Optimization (CoCoOp)
    """
    def __init__(self, args, classnames, clip_model):
        args.num_prompt = 1
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = CoCoOpPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image, labels=None, test=False):
        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[0] # only use the avg-pooled features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits
