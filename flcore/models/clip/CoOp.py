import torch
import math
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from .DenseCLIP import DenseCLIP
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

class DenseCoOpCLIP(DenseCLIP):
    """ Context Optimization (CoOp) for Dense Prediction
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        # args.num_prompt = 1 # CoOp uses only 1 prompt for each class
        self.prompt_learner = BasePromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        ignore_index = INFO[args.dataset]['ignore_index']
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.seg_text_loss_scale = args.seg_text_loss_scale

    def forward(self, image, labels=None, test=False):
        image_features = list(self.image_encoder(image.type(self.dtype)))
        # image_features = image_features[0] # only use the avg-pooled features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_embeds = image_features[5][1:]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        HW, B, D = image_embeds.shape
        H = W = int(math.sqrt(HW))
        image_embeds = image_embeds.permute(1,2,0) # B, D HW
        image_embeds = image_embeds.view(B, D, H, W)
        image_features[-1] = image_embeds

        # text
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.expand(image.shape[0], -1, -1) # [bs, cls, d]

        # segmentation decoder
        output = self.decoder(image_features) # use only last 5 featmaps
        # output = output / output.norm(dim=1, keepdim=True) # [bs, cls, h, w]

        score_map = torch.einsum('bchw, bkc->bkhw', image_embeds, text_features)
        self.cal_match_loss(score_map, labels)

        return output

    def add_loss(self, ):
        """ additional loss term for PromptSRC
        """
        return self.match_loss * self.seg_text_loss_scale

    def cal_match_loss(self, score_map, labels):
        """ score_map: B, D, H, W
            label: B, H, W
        """
        B, D, H, W = score_map.shape
        labels = F.interpolate(labels.unsqueeze(0).float(), size=(H, W), mode="nearest")
        self.match_loss = self.criterion(score_map, labels.squeeze(0).long())