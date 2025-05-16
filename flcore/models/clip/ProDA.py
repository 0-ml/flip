import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import ProDAPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO



class ProDACLIP(BaseCLIP):
    """ Prompt Distribution Learning
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.multi_label = False
        self.n_cls = len(classnames)
        self.prompt_learner = ProDAPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_prm = args.num_prompt
        self.ortho_loss_ratio = 1.

    def forward(self, image, labels=None, test=False):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features[0]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        n_class = self.n_cls

        if test:
            text_features = self.text_features
            # prompt, tokenized_prompts = self.prompt_learner(test=True)
            # text_features = self.text_encoder(prompt, tokenized_prompts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits
        else:
            assert labels is not None
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts  = self.prompt_learner()
            n_prompt = text_prompt.shape[0]//n_class

            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(n_class, n_prompt, -1)
            text_mean = text_features.mean(dim=1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_mean.t()

            batch_size = labels.shape[0]

            text_features = text_features - text_mean.unsqueeze(1)
            diag_cov_martix = text_features.permute(2,0,1) @ text_features.permute(2,1,0)
            diag_cov_martix /= n_prompt + 1
            refined_logits = torch.einsum("bd, dik -> bik", [image_features**2, diag_cov_martix])

            sigma = refined_logits[torch.arange(batch_size), labels, labels].unsqueeze(-1) + \
                refined_logits[:, torch.arange(n_class), torch.arange(n_class) ] - \
                2 * refined_logits[torch.arange(batch_size), labels, : ]

            logits += 0.5*(logit_scale**2)*sigma.view(-1, n_class)

            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            if n_prompt > 1:
                self.loss_m = dis[~torch.eye(self.n_prm, dtype=torch.bool, device=self.device)].abs().mean()
            else:
                self.loss_m = 0

            return logits

    def add_loss(self, ):
        """ diversity loss of ProDA
        """
        return self.loss_m * self.ortho_loss_ratio

    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner(test=True)
        try:
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
        except:
            text_features = []
            batch_size = 1000
            for bi in range(text_prompt.shape[0]//batch_size):
                batch_text_features = self.text_encoder(text_prompt[bi*1000:(bi+1)*1000], tokenized_prompts[bi*1000:(bi+1)*1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_cls, self.n_prm, -1)
        text_features = text_features.mean(dim=1)
        self.text_features = text_features
