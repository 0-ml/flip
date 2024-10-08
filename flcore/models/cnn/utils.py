import torch
from torch import nn
from .containers import (SequentialMultiOutput)

class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # yapf: disable
        stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.conv2,
            model.bn2,
            model.relu,
            model.conv3,
            model.bn3,
            model.relu,
            model.avgpool,
        )
        layers = [
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.attnpool
        ]

        # ResNet50 ImageNet Pretrain
        # stem = nn.Sequential(
        #     model.conv1,
        #     model.bn1,
        #     model.relu,
        #     model.maxpool,
        # )

        # # ViT-B/16
        # stem = nn.Sequential(
        #     model.conv1,
        #     model.ln_pre,
        # )
        # layers = [
        #     model.transformer,
        #     model.ln_post
        # ]

        self.m = SequentialMultiOutput(stem, *layers)

    def forward(self, x):
        return self.m(x)

class SelectOne(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        return xs[self.idx]

# ResNet50 ImageNet pretrained
# feat_shapes = {
#     'RN50':[torch.Size([1, 64, 56, 56]),
#             torch.Size([1, 64, 56, 56]),
#             torch.Size([1, 128, 28, 28]),
#             torch.Size([1, 256, 14, 14]),
#             torch.Size([1, 512, 7, 7])]
# }

# real outputs
# feat_shapes = {
#     'RN50':[torch.Size([1, 64, 56, 56]),
#             torch.Size([1, 256, 56, 56]),
#             torch.Size([1, 512, 28, 28]),
#             torch.Size([1, 1024, 14, 14]),
#             torch.Size([1, 2048, 7, 7]),
#             torch.Size([1, 1024, 7, 7]),
#             ]
# }

feat_shapes = {
    'RN50':[
                torch.Size([1, 64, 56, 56]),
                torch.Size([1, 256, 56, 56]),
                torch.Size([1, 512, 28, 28]),
                torch.Size([1, 1024, 14, 14]),
                torch.Size([1, 2048, 7, 7]),
                torch.Size([1, 1024, 7, 7])
            ],

    'RN101':[
                torch.Size([1, 64, 112, 112]),
                torch.Size([1, 256, 112, 112]),
                torch.Size([1, 512, 56, 56]),
                torch.Size([1, 1024, 28, 28]),
                torch.Size([1, 2048, 14, 14]),
                torch.Size([1, 512, 7, 7])
            ], # image shape:[3, 448, 448]

    'RN101':[
                torch.Size([1, 64, 160, 160]),
                torch.Size([1, 256, 160, 160]),
                torch.Size([1, 512, 80, 80]),
                torch.Size([1, 1024, 40, 40]),
                torch.Size([1, 2048, 20, 20]),
                torch.Size([1, 512, 7, 7])
            ], # image shape:[3, 640, 640]

    'ViT-B/16':[
                torch.Size([1, 256, 56, 56]),
                torch.Size([1, 512, 28, 28]),
                torch.Size([1, 1024, 14, 14]),
                # torch.Size([1, 2048, 7, 7]),
                # torch.Size([1, 1024, 7, 7])
            ]
}

# feat_shapes = {
#     'RN50':[torch.Size([32, 256, 56, 56]),
#             torch.Size([32, 512, 28, 28]),
#             torch.Size([32, 1024, 14, 14]),
#             torch.Size([32, 2048, 7, 7]),
#     ]
# }