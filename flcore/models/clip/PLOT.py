import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO



class PLOTCLIP(BaseCLIP):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.n_cls = len(classnames)
        self.image_size = INFO[args.dataset]['shape'][-1]
        self.prompt_learner = BasePromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_prm = args.num_prompt
        self.dataset = args.dataset
        self.use_uniform = True
        self.eps = 0.1
        self.max_iter = 100
        self.image_backbone = "ResNet" if 'RN' in args.image_backbone \
                                                        else 'ViT'

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def forward(self, image, labels=None, test=False):

        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        if self.image_backbone == 'ResNet':
            image_feature_pool = image_features[0]
            image_features = image_features[1:]
        else:
            image_feature_pool = image_features
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()
        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        if self.dataset == "ImageNet":
            text_features = self.text_encoder(prompts.to(self.cuda_device),
                                              tokenized_prompts.to(self.cuda_device))
            text_features = text_features.to(self.device)
            text_features =  text_features.contiguous().view(self.n_prm, self.n_cls, self.d)
            text_feature_pool = text_features.mean(dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features =  text_features.contiguous().view(self.n_prm, self.n_cls, self.d)
            text_feature_pool = text_features.mean(dim=0)

        image_features =  F.normalize(image_features, dim=2)
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(M,self.n_prm,b*self.n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim
        xx=torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*self.n_cls, self.n_prm, dtype=sim.dtype, device=sim.device).fill_(1. / self.n_prm)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()
        logits2 = logit_scale * sim_op
        if self.dataset == "ImageNet":
            logits2 = (logits2 + logits)
        return logits2
