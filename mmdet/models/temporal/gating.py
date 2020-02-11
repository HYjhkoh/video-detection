import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init


@TEMPORAL.register_module
class Gating(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super(Gating, self).__init__()
        self.in_channels = in_channels
        conv_previous    = []
        sigmoid_previous = []
        conv_present     = []
        sigmoid_present  = []
        conv             = []
        relu             = []
        for i in range(len(self.in_channels)):
            conv_previous.append(nn.Conv2d(in_channels[i]*2, 1, kernel_size=3, padding=1))
            sigmoid_previous.append(nn.Sigmoid())
            conv_present.append(nn.Conv2d(in_channels[i]*2,1,kernel_size=3,padding=1))
            sigmoid_present.append(nn.Sigmoid())

            conv.append(nn.Conv2d(in_channels[i]*2, in_channels[i], kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
        self.conv_previous = nn.ModuleList(conv_previous)
        self.sigmoid_previous = nn.ModuleList(sigmoid_previous)
        self.conv_present = nn.ModuleList(conv_present)
        self.sigmoid_present = nn.ModuleList(sigmoid_present)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x_pre, x):

        x_temp = []
        for feat1, feat2, conv_previous, sigmoid_previous, conv_present, sigmoid_present, conv, relu in zip(x_pre, x, self.conv_previous, self.sigmoid_previous, self.conv_present, self.sigmoid_present, self.conv, self.relu):
            
            feat_cat = torch.cat([feat1, feat2], 1)
            previous_weight = sigmoid_previous(conv_previous(feat_cat))
            present_weight = sigmoid_present(conv_present(feat_cat))
            feat_previous = feat1 * previous_weight
            feat_present = feat2 * present_weight

            feat_temp = torch.cat([feat_previous, feat_present],1)
            feat_temp = relu(conv(feat_temp))
            x_temp.append(feat_temp)

        return x_temp
