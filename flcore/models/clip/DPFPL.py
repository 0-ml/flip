import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner, DPFPLPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO






class DPFPLCLIP(BaseCLIP):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = DPFPLPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image, labels=None, test=False):
        image_features = self.image_encoder(image.type(self.dtype)) # [batch, 3, 224, 224] -> [32, 512]
        image_features = image_features[0]
        client_prompt = self.prompt_learner() # [100,77,512] = [n_cls, clip prompt token limit, ctx_dim]
        tokenized_prompts = self.tokenized_prompts
        client_text_features = self.text_encoder(client_prompt, tokenized_prompts) # [100,512] = [n_cls, ctx_dim]

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        client_text_features = client_text_features / client_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity between local text features and image features
        sim = image_features @ client_text_features.t() # [batch, n_cls]
        local_image_logits = sim * self.logit_scale.exp()

        return local_image_logits