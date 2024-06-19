import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO

class CoOpCLIP(BaseCLIP):
    """ Context Optimization (CoOp)
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        # args.num_prompt = 1 # CoOp uses only 1 prompt for each class
        self.prompt_learner = BasePromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image, labels=None, test=False):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[0] # only use the avg-pooled features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
