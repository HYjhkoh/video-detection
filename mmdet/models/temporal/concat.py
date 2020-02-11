import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init


@TEMPORAL.register_module
class Concat(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 seq_len=2):
        super(Concat, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        conv1 = []
        relu1 = []
        conv2 = []
        relu2 = []
        for i in range(len(self.in_channels)):
            conv1.append(
              nn.Conv2d(
                in_channels[i]*2,
                in_channels[i],
                kernel_size=3,
                padding=1))
            relu1.append(
              nn.ReLU(inplace=True))

            conv2.append(
              nn.Conv2d(
                in_channels[i],
                in_channels[i],
                kernel_size=3,
                padding=1))
            relu2.append(
              nn.ReLU(inplace=True))
        self.conv1 = nn.ModuleList(conv1)
        self.relu1 = nn.ModuleList(relu1)
        self.conv2 = nn.ModuleList(conv2)
        self.relu2 = nn.ModuleList(relu2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x_pre, x):

        x_temp = []
        for feat1, feat2, conv1, relu1, conv2, relu2 in zip(x_pre, x, self.conv1, self.relu1, self.conv2, self.relu2):
            feat_temp = torch.cat([feat1, feat2], 1)
            feat_temp = relu1(conv1(feat_temp))
            feat_temp = relu2(conv2(feat_temp))
            x_temp.append(feat_temp)

        return x_temp
