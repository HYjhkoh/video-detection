import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init


@TEMPORAL.register_module
class ConcatSplit(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 num_classes=31):
        super(ConcatSplit, self).__init__()
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        self.in_channels_cls = [num_anchors[i]*(num_classes) \
                                for i in range(len(in_channels))]
        self.in_channels_reg = [num_anchors[i]* 4 \
                                for i in range(len(in_channels))]
        conv1_cls = []
        relu1_cls = []
        conv1_reg = []
        relu1_reg = []
        conv2_cls = []
        relu2_cls = []
        conv2_reg = []
        relu2_reg = []

        for i in range(len(in_channels)):
            conv1_cls.append(
              nn.Conv2d(self.in_channels_cls[i]* 2, self.in_channels_cls[i],\
                        kernel_size=3,padding=1))
            relu1_cls.append(
              nn.ReLU(inplace=True))

            conv1_reg.append(
              nn.Conv2d(self.in_channels_reg[i]* 2, self.in_channels_reg[i],\
                        kernel_size=3,padding=1))
            relu1_reg.append(
              nn.ReLU(inplace=True))

            conv2_cls.append(
              nn.Conv2d(self.in_channels_cls[i], self.in_channels_cls[i],\
                        kernel_size=3, padding=1))
            relu2_cls.append(
              nn.ReLU(inplace=True))
            conv2_reg.append(
              nn.Conv2d(self.in_channels_reg[i], self.in_channels_reg[i], \
                        kernel_size=3, padding=1))
            relu2_reg.append(
              nn.ReLU(inplace=True))
        
        self.conv1_cls = nn.ModuleList(conv1_cls)
        self.relu1_cls = nn.ModuleList(relu1_cls)
        self.conv1_reg = nn.ModuleList(conv1_reg)
        self.relu1_reg = nn.ModuleList(relu1_reg)

        self.conv2_cls = nn.ModuleList(conv2_cls)
        self.relu2_cls = nn.ModuleList(relu2_cls)
        self.conv2_reg = nn.ModuleList(conv2_reg)
        self.relu2_reg = nn.ModuleList(relu2_reg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x_pre, x_cur):

        x_temp_cls = []
        x_temp_reg = []
        x_pre_cls = x_pre[0]
        x_pre_reg = x_pre[1]
        x_cur_cls = x_cur[0]
        x_cur_reg = x_cur[1]
        for feat1_cls, feat1_reg, feat2_cls, feat2_reg, conv1_cls, relu1_cls, conv1_reg, relu1_reg, conv2_cls, relu2_cls, conv2_reg, relu2_reg in zip(x_pre_cls, x_pre_reg, x_cur_cls, x_cur_reg, self.conv1_cls, self.relu1_cls, self.conv1_reg, self.relu1_reg, self.conv2_cls, self.relu2_cls, self.conv2_reg, self.relu2_reg):
            feat_temp_cls = torch.cat([feat1_cls, feat2_cls], 1)
            feat_temp_cls = relu1_cls(conv1_cls(feat_temp_cls))
            feat_temp_cls = relu2_cls(conv2_cls(feat_temp_cls))
            x_temp_cls.append(feat_temp_cls)

            feat_temp_reg = torch.cat([feat1_reg, feat2_reg], 1)
            feat_temp_reg = relu1_reg(conv1_reg(feat_temp_reg))
            feat_temp_reg = relu2_reg(conv2_reg(feat_temp_reg))
            x_temp_reg.append(feat_temp_reg)

        return (x_temp_cls, x_temp_reg)
