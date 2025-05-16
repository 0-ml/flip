import torch
import torch.nn as nn
from collections import OrderedDict

from ..clip import clip
from ..clip.clip import load_clip_to_cpu
from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ...datasets.info import INFO
from .template import IMAGENET_TEMPLATES
from ...utils import filter_states

_tokenizer = _Tokenizer()

class BasePromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.task = args.task
        self.prompt_algo = args.prompt_algo
        self.precision = args.precision
        self.n_cls = len(classnames)
        self.n_ctx = args.num_context
        self.n_prm = args.num_prompt
        self.embedding_func = clip_model.token_embedding
        self.ctx_init = args.ctx_init
        self.class_specific_context = args.class_specific_context == 'true'
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.vis_dim = clip_model.visual.output_dim
        self.image_backbone = args.image_backbone
        self.device = torch.device('cuda', args.device_id)
        clip_imsize = clip_model.visual.input_resolution
        image_size = INFO[args.dataset]['shape'][-1]
        cfg_imsize = image_size
        self.class_token_position = args.class_token_position
        if self.task == 'class':
            assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.prompt_prefix, self.ctx_vectors = self.init_prompt()

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx = nn.Parameter(self.ctx_vectors)  # n_prm, n_ctx, ctx_dim

        self.init_embedding(classnames)


    def init_embedding(self, classnames):
        self.n_cls = len(classnames) # update n_cls when classnames change
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        self.nc_prompts = [ self.prompt_prefix + '.' ]
        self.tokenized_prompts = self.init_tokenized_prompts(self.prompts)  # torch.Tensor
        with torch.no_grad():
            device = self.embedding_func.weight.device
            self.tokenized_prompts = self.tokenized_prompts.to(device)
            self.embedding = self.embedding_func(self.tokenized_prompts).type(self.dtype)
        self.registration()


    def init_tokenized_prompts(self, prompts):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        return tokenized_prompts.repeat(self.n_prm,1)

    def registration(self, ):
        # note: these are embedding vectors rather than tokens
        self.register_buffer("token_prefix", self.embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", self.embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS


    def init_prompt(self, ):
        if self.ctx_init:
            # use given words to initialize context vectors
            ctx_init = self.ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.embedding_func(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors = ctx_vectors.unsqueeze(0)
            prompt_prefix = ctx_init
        else:
            # random initialization
            if self.class_specific_context:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, self.n_ctx, self.ctx_dim,
                                          dtype=self.dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.n_prm, self.n_ctx, self.ctx_dim,
                                          dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        return prompt_prefix, ctx_vectors

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts

    def text_fixed_ebds(self, classnames):
        clip_model_temp = load_clip_to_cpu(self.image_backbone, True).float().to(self.device)
        all_teacher_features = []
        with torch.no_grad():
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.to(self.device))
                all_teacher_features.append(text_features.unsqueeze(1))
        text_feat_ebds = torch.cat(all_teacher_features,
                                               dim=1).mean(dim=1).to(self.device) # [n_cls, emb_dim]
        return text_feat_ebds / text_feat_ebds.norm(dim=-1, keepdim=True)


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.n_prm * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CoCoOpPromptLearner(BasePromptLearner):
    """ CoCoOp Prompt Learner
    """
    def __init__(self, args, classnames, clip_model):
        args.num_prompt = 1 # CoCoOp uses only 1 prompt for each class
        super().__init__(args, classnames, clip_model)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(self.vis_dim, self.vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(self.vis_dim // 16, self.ctx_dim))
        ]))

    def init_prompt(self,):
        if self.ctx_init:
            # use given words to initialize context vectors
            ctx_init = self.ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.embedding_func(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
        return prompt_prefix, ctx_vectors

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts) # [bs, n_cls, n_tkn, ctx_dim]

        return prompts


class ProDAPromptLearner(BasePromptLearner):
    """ ProDA Prompt Learner
    """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.device = torch.device('cuda', args.device_id)
        if self.n_prm >1:
            self.pos = [0 for _ in range(self.n_prm//4)] + [1 for _ in range(self.n_prm//4)] + [2 for _ in range(self.n_prm//2)]
        else:
            self.pos = [2 for _ in range(self.n_prm)]
        self.pos = torch.tensor(self.pos, device=self.device)
        self.iter_idx = 0
        self.prompt_bsz = args.prompt_batch_size
        assert self.n_prm % self.prompt_bsz == 0
        self.n_iter = int(self.n_prm / self.prompt_bsz)

    def init_tokenized_prompts(self, prompts):
        return torch.cat([clip.tokenize(p) for p in prompts])

    def registration(self, ):
        self.register_buffer('token_prefix', self.embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer('token_suffix', self.embedding[:, 1+self.n_ctx:, :]) # CLS, EOS, [n_cls, -1, ctx_dim]

        nc_tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts

        with torch.no_grad():
            device = self.embedding_func.weight.device
            self.nc_tokenized_prompts = self.nc_tokenized_prompts.to(device)
            embedding = self.embedding_func(self.nc_tokenized_prompts).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer('nc_token_suffix', embedding[:, 1+self.n_ctx:, :]) # EOS, [n_cls, -1, ctx_dim]


    def forward(self, test=False):
        if self.n_iter > 1 and (not test):
            if self.iter_idx == 0:
                self.select_idx = torch.randperm(self.n_prm, device=self.device)
            batch_idx = self.select_idx[self.iter_idx * self.prompt_bsz: (self.iter_idx+1) * self.prompt_bsz]
            ctx = self.ctx[batch_idx]
            pos = self.pos[batch_idx]

            self.iter_idx += 1
            if self.iter_idx == self.n_iter:
                self.iter_idx = 0
        else:
            ctx = self.ctx
            pos = self.pos

        prompts, tokenized_prompts = self.construct_prompts(ctx, pos)
        if test:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def construct_prompts_proda(self, ctx, pos):

        prompt_size = ctx.shape[0]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls*prompt_size, -1)

        ctx_end = ctx[pos==2]
        n_end = ctx_end.shape[0]
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1)
        ctx_end = ctx_end.unsqueeze(0).repeat(self.n_cls, 1, 1, 1)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2)

        ctx_middle = ctx[pos==1]
        n_middle = ctx_middle.shape[0]
        prompts_middle = []
        half_n_ctx = self.n_ctx // 2
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            ctx_i_half1 = ctx_middle[:, :half_n_ctx, :].unsqueeze(0)
            ctx_i_half2 = ctx_middle[:, half_n_ctx:, :].unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_middle, 1, dim)
                ctx_i_half1, # (1, n_middle, n_ctx//2, dim)
                class_i, # (1, n_middle, name_len, dim)
                ctx_i_half2, # (1, n_middle, n_ctx//2, dim)
                suffix_i # (1, n_middle, *, dim)
            ], dim=2)
            prompts_middle.append(prompt)
        prompts_middle = torch.cat(prompts_middle, dim=0)

        ctx_front = ctx[pos==0]
        n_front = ctx_front.shape[0]
        prompts_front = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            ctx_i = ctx_front.unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_front, 1, dim)
                class_i, # (1, n_front, name_len, dim)
                ctx_i, # (1, n_front, n_ctx, dim)
                suffix_i # (1, n_front, *, dim)
            ], dim=2)
            prompts_front.append(prompt)
        prompts_front = torch.cat(prompts_front, dim=0)
        prompts = torch.cat([prompts_end, prompts_middle, prompts_front], dim=1).view(prompt_size*self.n_cls, -1, self.ctx_dim)

        prompts = prompts_end.squeeze(dim=1)
        return prompts, tokenized_prompts


    def construct_prompts(self, ctx, pos):

        prompt_size = ctx.shape[0]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls*prompt_size, -1)

        ctx_end = ctx[pos==2]
        n_end = ctx_end.shape[0]
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1)
        ctx_end = ctx_end.unsqueeze(0).repeat(self.n_cls, 1, 1, 1)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2)
        prompts = prompts_end.squeeze(dim=1)
        return prompts, tokenized_prompts


    def only_prefix(self):
        ctx = self.ctx
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts


class VLPromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        self.n_ctx_vis = args.num_prompt_vision
        self.image_backbone = args.image_backbone

        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(self.image_backbone, True).float().to(self.device)
        clip_model_temp_image = load_clip_to_cpu(self.image_backbone, True)
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.to(self.device))
                all_teacher_features.append(text_features.unsqueeze(1))

        self.text_fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)

    def init_prompt(self, ):
        if self.ctx_init and self.n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.embedding_func(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
        return prompt_prefix, ctx_vectors

    def init_tokenized_prompts(self, prompts):
        return torch.cat([clip.tokenize(p) for p in prompts])

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

### Bayesian Prompt Learning for Image-Language Model Generalization ###

class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        out = self.module(inps)
        return out


class Amortized(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Amortized, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, weight_log_variance

class BPLPromptLearner(BasePromptLearner):
    """ Bayesian Prompt Learning
    """
    def __init__(self, args, classnames, clip_model):
        args.num_prompt = 1 # BPL only uses 1 learnable prompt
        super().__init__(args, classnames, clip_model)

        self.L = 4
        self.vpt_type = "cocoopvpt"

        if self.vpt_type == "cocoopvpt":
            self.meta_net = Amortized(
                input_units=self.vis_dim,
                d_theta=self.vis_dim // 2,
                output_units=self.ctx_dim
            )
            if self.precision == "fp16":
                self.meta_net.half()
        elif self.vpt_type == "coopvpt":
            self.mean_posterior = nn.Parameter(torch.zeros(1, self.ctx_dim, dtype=self.dtype))
            self.std_posterior = nn.Parameter(torch.rand(1, self.ctx_dim, dtype=self.dtype))
        else:
            raise ValueError(f"Type {self.vpt_type} is not supported.")


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def sample(self, mu, logvar, L):
        shape = (L,) + mu.size()
        eps = torch.randn(shape).type_as(mu)
        bias = mu.unsqueeze(0) + eps * logvar.exp().sqrt().unsqueeze(0)
        return bias

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        if self.vpt_type == "cocoopvpt":
            bias_mu, bias_logvar = self.meta_net(im_features.detach())  # (1, ctx_dim)
        elif self.vpt_type == "coopvpt":
            bias_mu, bias_logvar = self.mean_posterior, self.std_posterior  # (1, ctx_dim)
        else:
            raise ValueError(f"Type {self.vpt_type} is not supported.")

        bias = self.sample(bias_mu, bias_logvar, self.L)  # (L, 1, ctx_dim)
        # ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (L, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts, bias_mu, bias_logvar




CUSTOM_TEMPLATES = {
    "oxford_pets": "a photo of a {}, a type of pet.",
    "oxford_flowers": "a photo of a {}, a type of flower.",
    "fgvc_aircraft": "a photo of a {}, a type of aircraft.",
    "dtd": "a photo of a {}, a type of texture.",
    "eurosat": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "stanford_cars": "a photo of a {}.",
    "food101": "a photo of {}, a type of food.",
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
    "inaturalist": "a photo of a {}.",
}

class KgCoOpPromptLearner(BasePromptLearner):
    """ Visual-Language Prompt Tuning with Knowledge-guided Context Optimization
    """
    def __init__(self, args, classnames, clip_model):
        args.num_prompt = 1
        super().__init__(args, classnames, clip_model)

        n_cls = len(classnames)
        bias_vectors = torch.empty(1, 512, dtype=self.dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        #print(f"Loading CLIP (backbone: {args.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(args.image_backbone)
        clip_model_.to(self.device)

        # get text features of zs clip model
        temp = CUSTOM_TEMPLATES[args.dataset]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.to(self.device)
        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512)),
            ("relu", nn.ReLU(inplace=True))
            #("linear2", nn.Linear(128, 512))
        ]))

        if self.precision == "fp16":
            self.meta_net.half()

    def init_tokenized_prompts(self, prompts):
        return torch.cat([clip.tokenize(p) for p in prompts])

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.n_prm * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts




class PGPPromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.bottleneck = args.pgp_bottleneck

        if self.ctx_init:
            # use given words to initialize context vectors
            ctx_init = self.ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if self.class_specific_context:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, self.n_ctx, self.ctx_dim, dtype=self.dtype)
            else:
                print("Initializing a generic context")
                U = torch.empty(self.n_prm, self.n_ctx, self.bottleneck, dtype=self.dtype)
                V = torch.empty(self.n_prm, self.bottleneck, self.ctx_dim, dtype=self.dtype)
                sigma = torch.empty(self.n_prm, self.n_ctx, self.ctx_dim, dtype = self.dtype)

            nn.init.normal_(U, std=0.02)
            nn.init.normal_(V, std=0.02)
            nn.init.normal_(sigma, std=0.02)# define the prompt to be trained
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self.sigma = nn.Parameter(sigma)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.name_lens = name_lens

    def forward(self):

        # ctx = self.ctx
        U = self.U
        V = self.V
        UV = torch.matmul(U,V)
        sigma = self.sigma
        ctx = UV +self.sigma
        embedding = self.embedding

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.n_prm * self.n_cls, self.n_ctx, ctx.shape[3])

        if UV.dim() == 3:
            UV = UV.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        UV = UV.permute(1, 0, 2, 3)
        UV = UV.contiguous().view(self.n_prm * self.n_cls, self.n_ctx, UV.shape[3])

        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        sigma = sigma.permute(1, 0, 2, 3)
        sigma = sigma.contiguous().view(self.n_prm * self.n_cls, self.n_ctx, sigma.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_sigma = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    sigma,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_UV = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    UV,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return  embedding, prompts_sigma, prompts_UV, prompts


from .utils import PromptTranslator
class TPGPromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)

        self.ctx_depth = 1
        self.meta_net = PromptTranslator(self.n_ctx, self.ctx_depth, depth=0)
        if self.precision == "fp16":
            self.meta_net.half()


    def forward(self, context_emb):
        text_ctx, vis_ctx = self.meta_net(context_emb.unsqueeze(0))  # (n_ctx, ctx_dim) # self.ctx

        return text_ctx, vis_ctx


class FolioPromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)


    def forward(self):
        ctx = self.ctx
        if not self.class_specific_context:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            ctx = ctx.permute(1, 0, 2, 3)
            ctx = ctx.contiguous().view(self.n_prm*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


def orthogonalize(matrix):
    m = matrix.shape[1]
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col

def factorize_ctx(origin, rank, device):


    with torch.no_grad():
        v = torch.normal(0, 1, size=(origin.shape[1], rank)).type(origin.dtype) # [ctx_dim, rank]
        u = torch.matmul(origin.to(device), v.to(device)) # [n_ctx, rank]
        orthogonalize(u)
        v = torch.matmul(origin.t().to(device), u.to(device)) # [ctx_dim, rank]
        orthogonalize(v)
        v = v.t() # [rank, ctx_dim]
        residual = origin.to(device) - torch.matmul(u.to(device), v.to(device)) # [n_ctx, ctx_dim]

    return (u, v, residual)

class DPFPLPromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.factorization = 'dpfpl'
        self.rank = args.dpfpl_rank

        # global context vector
        global_ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype) # n_ctx = 16, ctx_dim = 512
        nn.init.normal_(global_ctx_vectors, std=0.02)
        self.global_ctx = nn.Parameter(global_ctx_vectors)

        # local u and v context vectors
        if self.factorization in ['dpfpl']:
            local_ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype) # n_ctx = 16, ctx_dim = 512
            nn.init.normal_(local_ctx_vectors, std=0.02)
            self.local_ctx = nn.Parameter(local_ctx_vectors)
        if self.factorization in ['dpfpl']:
            local_u_ctx_vectors = torch.empty(self.n_ctx, self.rank, dtype=self.dtype)
            nn.init.normal_(local_u_ctx_vectors, std=0.02)
            self.local_u_ctx = nn.Parameter(local_u_ctx_vectors)
            local_v_ctx_vectors = torch.empty(self.rank, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(local_v_ctx_vectors, std=0.02)
            self.local_v_ctx = nn.Parameter(local_v_ctx_vectors)

    def forward(self):
        if self.factorization == 'promptfl':
            client_ctx = self.global_ctx
        elif self.factorization == 'fedotp':
            client_ctx = self.global_ctx + self.local_ctx
        elif self.factorization == 'fedpgp':
            client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx)
        else:
            local_u_ctx, local_v_ctx, residual = factorize_ctx( self.local_ctx.data,
                                                                self.rank,
                                                                self.device)
            self.local_u_ctx.data = local_u_ctx
            self.local_v_ctx.data = local_v_ctx
            if self.factorization == 'dplora':
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx)
            elif self.factorization == 'dpfpl':
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx) + residual

        if client_ctx.dim() == 2:
            client_ctx = client_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        client_prompt = torch.cat(
            [
                self.token_prefix,
                client_ctx,
                self.token_suffix,
            ],
            dim=1,
        )

        return client_prompt


from .utils import _get_clones
class MaplePromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        # Default is 1, which is compound shallow prompting
        assert args.prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.prompt_depth # max=12, but will create 11 such shared prompts

        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(self.ctx_dim, 768)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(self.ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)



    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.n_prm * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        # pass here original, as for visual 768 is required
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts



class DensePromptLearner(BasePromptLearner):
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
