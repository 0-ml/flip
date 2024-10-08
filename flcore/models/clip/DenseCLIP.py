import torch
import torch.nn as nn
import torch.nn.functional as F
from ..text.encoder import TextEncoder
from ..cnn.utils import ResNetFeatureMapsExtractor, feat_shapes, SelectOne
from ..cnn.fpn import FPN, PanopticFPN
from ..cnn.layers import Interpolate
from ...datasets.info import INFO
from .base import BaseCLIP

from torchvision.models import resnet50, ResNet50_Weights

class DenseCLIP(BaseCLIP):
    """ Base Dense CLIP class """
    def __init__(self, args, classnames, clip_model):
        super().__init__(args, classnames, clip_model)
        self.img_size = INFO[args.dataset]['shape'][1:]
        self.num_classes = INFO[args.dataset]['num_classes']
        # self.image_encoder = clip_model.visual
        self.image_encoder = ResNetFeatureMapsExtractor(clip_model.visual)
        # self.image_encoder = ResNetFeatureMapsExtractor(resnet50(
        #                                 weights=ResNet50_Weights.IMAGENET1K_V1))
        self.decoder = nn.Sequential(
                PanopticFPN(feat_shapes[args.image_backbone],
                    hidden_channels=256,
                    # out_channels=self.num_classes),
                    out_channels=256),
                # SelectOne(idx=0),
                # Interpolate(size=self.img_size),
                # upsampling
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=self.num_classes,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    dilation=1,
                    bias=True
                )
            )
