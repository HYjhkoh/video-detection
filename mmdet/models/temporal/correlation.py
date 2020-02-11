import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init
from spatial_correlation_sampler import spatial_correlation_sample


@TEMPORAL.register_module
class Correlation(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super(Correlation, self).__init__()
        self.in_channels = in_channels

        leaky_relu = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(
              nn.LeakyReLU(inplace=True))
        self.leaky_relu = nn.ModuleList(leaky_relu)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x_pre, x):

        x_move = []
        for feat1, feat2, leaky_relu, channels in zip(x, x_pre, self.leaky_relu, \
                                                      self.in_channels):
            corr_feat = spatial_correlation_sample(feat1,
                                                   feat2,
                                                   kernel_size=1,
                                                   patch_size=3,
                                                   stride=1,
                                                   padding=0,
                                                   dilation_patch=1)

            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat/channels)
            x_move.append(corr_feat)

        return x_move
