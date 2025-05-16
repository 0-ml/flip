import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner, FolioPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO
from ..clip import clip




class PromptFolioCLIP(BaseCLIP):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = FolioPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_cls = len(classnames)
        self.frac = args.folio_frac

    def forward(self, image, labels=None, test=False):

        image_features = self.get_img_features(image)
        image_features = image_features[0]
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        # feature0 is global, feature1 is local
        text_features0 = self.text_encoder(prompts[:self.n_cls], tokenized_prompts[:self.n_cls])
        text_features1 = self.text_encoder(prompts[self.n_cls:2 * self.n_cls], tokenized_prompts[self.n_cls:2 * self.n_cls])



        text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # frac = 0 means fully global
        # frac = 1 means fully local
        text_features = (1 - self.frac) * text_features0 + self.frac * text_features1
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_img_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features