import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO
from ..clip import clip

CUSTOM_TEMPLATES = {
    "oxford_pets": "a type of pet, a photo of a {}.",
    "oxford_flowers": "a type of flower, a photo of a {}.",
    "fgvc_aircraft": "a type of aircraft, a photo of a {}.",
    "dtd": "a texture of {}.",
    "eurosat": "a centered satellite photo of {}.",
    "stanford_cars": "a photo of a {}.",
    "food101": "a type of food, a photo of {}.",
    "sun397": "a photo of a {}.",
    "caltech101": "a photo of a {}.",
    "ucf": "a photo of a person doing {}.",
    "imagenet": "a photo of a {}.",
    "imagenet_s": "a photo of a {}.",
    "imagenetv2": "a photo of a {}.",
    "imagenet_a": "a photo of a {}.",
    "imagenet_r": "a photo of a {}.",
    "imagenet_s": "a photo of a {}.",
    "domain_net": "a photo of a {}.",
    "tiny_imagenet": "a photo of a {}.",
}

class ZSCLIP(BaseCLIP):
    """ zero shot CLIP model
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.n_prm = args.num_prompt
        temp = CUSTOM_TEMPLATES[args.dataset]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.n_prm,1)

        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features[0] # only use pooled feature
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits