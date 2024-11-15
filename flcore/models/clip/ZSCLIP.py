import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO
from ..clip import clip
from .utils import CUSTOM_TEMPLATES

class CLIPPromptLearner:
    def __init__(self, args, classnames, clip_model):
        # super().__init__()
        self.args = args
        self.n_prm = args.num_prompt
        self.classnames = classnames
        self.dataset = args.dataset
        self.clip_model = clip_model
        self.device = torch.device('cuda', args.device_id)


    def init_embedding(self, classnames):
        self.clip_model = self.clip_model.to('cpu')
        temp = CUSTOM_TEMPLATES[self.dataset]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.n_prm,1)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
        self.text_features = text_features
        self.clip_model = self.clip_model.to(self.device)

class ZSCLIP(BaseCLIP):
    """ zero shot CLIP model
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.clip_model = clip_model
        self.prompt_learner = CLIPPromptLearner(args, classnames, clip_model)
        self.prompt_learner.init_embedding(classnames)


    def forward(self, image, labels=None, test=False):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features[0] # only use pooled feature
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.prompt_learner.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits
