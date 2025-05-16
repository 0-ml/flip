import torch
import torch.nn as nn
from torch.nn import functional as F

from ..text.prompt import BasePromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO

class BaseCLIP(nn.Module):
    """ Base CLIP class """
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device('cuda', args.device_id)
        self.cuda_device = torch.device("cuda")
        self.batch_size = args.batch_size

    def forward(self,):
        raise NotImplementedError

    @torch.no_grad()
    def set_classifier(self):
        pass

    def add_loss(self,):
        """ additional loss
        """
        return 0

    def update_global_text_feats(self,):
        pass

    def custom_avg(self, *args, **kwargs):
        pass