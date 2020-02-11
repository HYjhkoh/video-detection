import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init


@TEMPORAL.register_module
class GatingSplit(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 num_classes=31):
        super(GatingSplit, self).__init__()
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        self.in_channels_cls = [num_anchors[i]*(num_classes) \
                                for i in range(len(in_channels))]
        self.in_channels_reg = [num_anchors[i]* 4 \
                                for i in range(len(in_channels))]
        conv_previous_cls    = []
        sigmoid_previous_cls = []
        conv_present_cls     = []
        sigmoid_present_cls  = []
        conv_cls             = []
        relu_cls             = []

        conv_previous_reg    = []
        sigmoid_previous_reg = []
        conv_present_reg     = []
        sigmoid_present_reg  = []
        conv_reg             = []
        relu_reg             = []

        for i in range(len(in_channels)):
            conv_previous_cls.append(nn.Conv2d(self.in_channels_cls[i]* 2, 1,\
                                               kernel_size=3, padding=1))
            sigmoid_previous_cls.append(nn.Sigmoid())
            conv_present_cls.append(nn.Conv2d(self.in_channels_cls[i]* 2, 1,\
                                              kernel_size=3, padding=1))
            sigmoid_present_cls.append(nn.Sigmoid())

            conv_cls.append(nn.Conv2d(self.in_channels_cls[i]* 2, self.in_channels_cls[i],\
                                      kernel_size=3, padding=1))
            relu_cls.append(nn.ReLU(inplace=True))

            conv_previous_reg.append(nn.Conv2d(self.in_channels_reg[i]* 2, 1,\
                                               kernel_size=3, padding=1))
            sigmoid_previous_reg.append(nn.Sigmoid())
            conv_present_reg.append(nn.Conv2d(self.in_channels_reg[i]* 2, 1,\
                                              kernel_size=3, padding=1))
            sigmoid_present_reg.append(nn.Sigmoid())

            conv_reg.append(nn.Conv2d(self.in_channels_reg[i]* 2, self.in_channels_reg[i], \
                                      kernel_size=3, padding=1))
            relu_reg.append(nn.ReLU(inplace=True))

        self.conv_previous_cls = nn.ModuleList(conv_previous_cls)
        self.sigmoid_previous_cls = nn.ModuleList(sigmoid_previous_cls)
        self.conv_present_cls = nn.ModuleList(conv_present_cls)
        self.sigmoid_present_cls = nn.ModuleList(sigmoid_present_cls)
        self.conv_cls = nn.ModuleList(conv_cls)
        self.relu_cls = nn.ModuleList(relu_cls)

        self.conv_previous_reg = nn.ModuleList(conv_previous_reg)
        self.sigmoid_previous_reg = nn.ModuleList(sigmoid_previous_reg)
        self.conv_present_reg = nn.ModuleList(conv_present_reg)
        self.sigmoid_present_reg = nn.ModuleList(sigmoid_present_reg)
        self.conv_reg = nn.ModuleList(conv_reg)
        self.relu_reg = nn.ModuleList(relu_reg)

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
        for feat1_cls, feat1_reg, feat2_cls, feat2_reg, conv_previous_cls, sigmoid_previous_cls, conv_previous_reg, sigmoid_previous_reg, conv_present_cls, sigmoid_present_cls, conv_present_reg, sigmoid_present_reg, conv_cls, relu_cls, conv_reg, relu_reg in zip(x_pre_cls, x_pre_reg, x_cur_cls, x_cur_reg, self.conv_previous_cls, self.sigmoid_previous_cls, self.conv_previous_reg, self.sigmoid_previous_reg, self.conv_present_cls, self.sigmoid_present_cls, self.conv_present_reg, self.sigmoid_present_reg, self.conv_cls, self.relu_cls, self.conv_reg, self.relu_reg):

            feat_cat_cls = torch.cat([feat1_cls, feat2_cls], 1)
            previous_weight_cls = sigmoid_previous_cls(conv_previous_cls(feat_cat_cls))
            present_weight_cls = sigmoid_present_cls(conv_present_cls(feat_cat_cls))
            feat_previous_cls = feat1_cls * previous_weight_cls
            feat_present_cls = feat2_cls * present_weight_cls

            feat_temp_cls = torch.cat([feat_previous_cls, feat_present_cls],1)
            feat_temp_cls = relu_cls(conv_cls(feat_temp_cls))
            x_temp_cls.append(feat_temp_cls)

            feat_cat_reg = torch.cat([feat1_reg, feat2_reg], 1)
            previous_weight_reg = sigmoid_previous_reg(conv_previous_reg(feat_cat_reg))
            present_weight_reg = sigmoid_present_reg(conv_present_reg(feat_cat_reg))
            feat_previous_reg = feat1_reg * previous_weight_reg
            feat_present_reg = feat2_reg * present_weight_reg

            feat_temp_reg = torch.cat([feat_previous_reg, feat_present_reg],1)
            feat_temp_reg = relu_reg(conv_reg(feat_temp_reg))
            x_temp_reg.append(feat_temp_reg)

        return x_temp_cls, x_temp_reg
