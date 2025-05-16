import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque

from .base import BaseCLIP
from ..text.prompt import NPPromptLearner
from ..text.encoder import TextEncoder
from ...datasets.info import INFO
from ..neural_process.np_head import NP_HEAD
from ...utils import calculate_js_loss, reg_loss


class NPCLIP(BaseCLIP):
    """ Neural Process CLIP
    """
    def __init__(self, args, classnames, clip_model, ):
        super().__init__(args, classnames, clip_model, )
        self.prompt_learner = NPPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.n_cls = len(classnames)
        self.js_loss_ratio = 0.
        self.p_reg_loss_ratio = 0.
        self.g_reg_loss_ratio = 0.
        self.g_forward_times = args.global_forward_times
        self.l_forward_times = args.local_forward_times
        self.global_text_feats = deque(maxlen=5)


    def update_global_text_feats(self,):
        with torch.no_grad():
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            ctx = self.prompt_learner.ctx.expand(self.n_cls, -1, -1)
            prefix = self.prompt_learner.token_prefix
            suffix = self.prompt_learner.token_suffix
            prompts = self.prompt_learner.construct_prompts(ctx, prefix, suffix)
            text_feat = self.text_encoder(prompts, tokenized_prompts)
            self.global_text_feats.appendleft(text_feat)


    def forward(self, image, labels=None, test=False):
        # tokenized_prompts = self.tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[0] # only use the avg-pooled features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if test:
            bias = self.prompt_learner.np_head(image_features, phase_train=False,
                                               g_forward_times=self.g_forward_times,
                                               l_forward_times=self.l_forward_times)
        else:
            # split contex and target data points
            x_context, x_target = image_features.chunk(2)
            y_onehot = F.one_hot(labels, num_classes=self.n_cls)
            y_context, y_target = y_onehot.chunk(2)
            bias_context, mean_context, sigma_context = \
                self.prompt_learner.np_head(x_context, x_context_in=x_context,
                                     labels_in=y_context, labels_context_in=y_context,
                                     phase_train=True, g_forward_times=self.g_forward_times,
                                     l_forward_times=self.l_forward_times)

            bias_target, mean_target, sigma_target = \
                self.prompt_learner.np_head(x_target, x_context_in=x_context,
                                     labels_in=y_target, labels_context_in=y_context,
                                     phase_train=True, g_forward_times=self.g_forward_times,
                                     l_forward_times=self.l_forward_times)
            # calculate prompts
            bias = torch.cat((bias_context, bias_target), dim=1) # [L, bs, ctx_dim]

        prompts_shifted = self.prompt_learner(bias) # [L, bs, n_cls, len, ctx_dim]

        all_logits, all_text_feat = [], []
        for i, img_feat in enumerate(image_features):
            logits, text_feats = [], []
            img_feat = img_feat.unsqueeze(0)
            prompts = prompts_shifted[:, i, :, :, :]
            for prompt in prompts:
                text_feat = self.text_encoder(prompt, tokenized_prompts)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                text_feats.append(text_feat)
                logit = logit_scale * img_feat @ text_feat.t()
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
            text_feats = torch.stack(text_feats, dim=0)
            all_logits.append(logits)
            all_text_feat.append(text_feats)
        all_logits = torch.stack(all_logits, dim=0) # [bs, L, n_cls]
        all_logits = all_logits.permute(1,0,2) # [L, bs, n_cls]

        if not test:
            logits_context, logits_target = all_logits.chunk(2, dim=1)
            all_text_feat = torch.stack(all_text_feat, dim=0) # [bs, L, n_cls, dim]
            all_text_feat = all_text_feat.permute(1,0,2,3) # [L, bs, n_cls, dim]
            text_feat_context, text_feat_target = all_text_feat.chunk(2, dim=1)
            text_feat_context = text_feat_context.mean(dim=(0,1))

            self.js_loss = calculate_js_loss(logits_context, logits_target, mean_context,
                                          sigma_context, mean_target, sigma_target)
            # text feature regularization
            self.p_reg_loss = reg_loss(text_feat_context,
                                     self.prompt_learner.text_fixed_embeddings)
            # FL global regularization
            self.g_reg_loss = self.global_reg_loss(image_features,
                                                   all_logits.mean(0), logit_scale)
        return all_logits


    def global_reg_loss(self, img_features, local_logits, logit_scale):
        if len(self.global_text_feats) == 0:
            return 0
        else:
            global_text_feats = torch.stack(tuple(self.global_text_feats))
            text_features = torch.mean(global_text_feats, dim=0)
            global_logits = logit_scale * img_features @ text_features.t()
            kl_loss = nn.KLDivLoss()
            return kl_loss(global_logits, local_logits)


    def add_loss(self,):
        return self.js_loss * self.js_loss_ratio + \
               self.p_reg_loss * self.p_reg_loss_ratio + \
               self.g_reg_loss * self.g_reg_loss_ratio
