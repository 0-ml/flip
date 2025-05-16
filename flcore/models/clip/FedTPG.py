import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

from .base import BaseCLIP
from ..text.prompt import BasePromptLearner, TPGPromptLearner
from ...datasets.info import INFO
from ..clip import clip


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.forward_tpg(x, text_ctx, True)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class FedTPGCLIP(BaseCLIP):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.args = args
        self.prompt_learner = TPGPromptLearner(args, classnames, clip_model)
        self.set_prompt_prefix()

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.token_embedding = clip_model.token_embedding
        self.clip_model_ = clip_model


    def set_prompt_prefix(self):

        # random initialization
        self.prompt_prefix = " ".join(["X"] * self.prompt_learner.n_ctx)
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.prompt_learner.n_ctx}")


    def get_tokenized_classnames(self, classnames):

        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return embedding, tokenized_prompts

    def forward(self, image, labels=None, test=False):
        classnames = copy.deepcopy(self.prompt_learner.classnames)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts_ = classnames
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.to(self.device)

        with torch.no_grad():
            text_features_ = self.clip_model_.encode_text(prompts_)
            text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        """
        classnames = copy.deepcopy(self.prompt_learner.classnames)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_ = text_features / text_features.norm(dim=-1, keepdim=True)
        """

        text_features, vis_ctx = self.encode_text(classnames, text_features_)
        image_features = self.encode_image(image, vis_ctx)
        image_features = image_features[0] # only use the avg-pooled features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits

    def encode_image(self, image, vis_ctx):
        return self.image_encoder(image.type(self.dtype))

    def encode_text(self, classnames, text_features_):

        context_emb = text_features_
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classnames)

        text_ctx, vis_ctx = self.prompt_learner(context_emb)

        prompt_vectors = torch.cat(
            [
                prompt_vectors[:, :1],  # (dim0, 1, dim)
                text_ctx[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),  # (dim0, n_ctx, dim)
                prompt_vectors[:, 1 + text_ctx.shape[1]:],  # (dim0, *, dim)
            ],
            dim=1,
        )
        if len(text_ctx) > 1:
            text_ctx = text_ctx[1:]
        else:
            text_ctx = []
        text_features = self.text_encoder(prompt_vectors, tokenized_prompts, text_ctx)
        return text_features, vis_ctx