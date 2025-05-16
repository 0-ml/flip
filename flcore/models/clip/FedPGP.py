import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner, PGPPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO

class FedPGPCLIP(BaseCLIP):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = PGPPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.temp = 1
        self.contrastive_loss_ratio = 1

    def forward(self, image, labels=None, test=False):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding, prompts_sigma, prompts_UV, prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if test == False:
            text_features_0 = self.text_encoder(embedding.to(self.device), tokenized_prompts)
            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
            text_features_UV = self.text_encoder(prompts_UV, tokenized_prompts)

            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)
            text_features_UV = text_features_UV / text_features_UV.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            # contrastive loss
            cos = torch.nn.CosineSimilarity(dim=-1)
            posi = cos(text_features_0, text_features_sigma)
            nega = cos(text_features_sigma, text_features)

            logits_con = torch.cat((posi.reshape(-1, 1), nega.reshape(-1, 1)), dim=1)
            logits_con /= self.temp
            target = torch.zeros(logits_con.size(0)).to(self.device).long()
            self.contrastive_loss = F.cross_entropy(logits_con, target)

            return logits
        else:
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        return logits

    def add_loss(self, ):

        return self.contrastive_loss * self.contrastive_loss_ratio