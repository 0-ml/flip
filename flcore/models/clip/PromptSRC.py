import torch
import torch.nn as nn
import numpy as np
import copy
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import VLPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO
from ...utils import filter_states


class PromptSRCCLIP(BaseCLIP):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.prompt_learner = VLPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_cls = len(classnames)
        self.text_loss_weight = 25.
        self.image_loss_weight = 10.
        self.global_rounds = args.global_rounds
        self.prompt_algo = args.prompt_algo
        mean = 15
        stdev = 1
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(0, self.global_rounds + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.previous_model_gpa = None

    def forward(self, image, labels=None, test=False):
        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[0] # only use pooled feature
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if not test:
            # Now calculate the frozen pre-trained features
            text_fixed_ebds = self.prompt_learner.text_fixed_embeddings  # precomputed pre-trained frozen textual features
            text_fixed_ebds = text_fixed_ebds / text_fixed_ebds.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                image_fixed_ebds = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                image_fixed_ebds = image_fixed_ebds[0] # only use the pooled image embeddings
                image_fixed_ebds = image_fixed_ebds / image_fixed_ebds.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * image_fixed_ebds.to(self.device) @ text_fixed_ebds.half().to(self.device).t()
            # L_SCL_text loss
            loss_scl_text = F.l1_loss(text_features, text_fixed_ebds.to(self.device),
                                      reduction='mean') * self.text_loss_weight
            # L_SCL_image loss
            loss_scl_image = F.l1_loss(image_features, image_fixed_ebds.to(self.device),
                                       reduction='mean') * self.image_loss_weight
            # L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            self.L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
        # logits = logits.mean(dim=0)
        return logits


    def add_loss(self, ):
        """ additional loss term for PromptSRC
        """
        return self.L_SCL

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def custom_avg(self, cur_rounds):
        self.gauss_avg(cur_rounds)

    def gauss_avg(self, cur_rounds):
        current_epoch_weight = self.gauss[cur_rounds]
        states = {
                k: v.detach().clone().cpu()
                for k, v in self.prompt_learner.state_dict().items()
                                if filter_states(k)}
        current_model_weights = states
        weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
        if self.previous_model_gpa is None:
            self.previous_model_gpa = weighted_state_dict
        else:
            self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)
        self.prompt_learner.load_state_dict(self.previous_model_gpa, strict=False)

    def state_dict_weighting(self, main_dict, weightage):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        for k,v in main_dict.items():
            updated_dict[k] = v * weightage
        return updated_dict

    def state_dict_add(self, dict1, dict2):
        # Average all parameters
        modified_dict = dict2
        for k,v in dict1.items():
            modified_dict[k] = (modified_dict[k] + dict1[k])
        return modified_dict