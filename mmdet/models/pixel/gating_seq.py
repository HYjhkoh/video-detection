import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init


@TEMPORAL.register_module
class GatingSeq(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(GatingSeq, self).__init__()
        self.in_channels    = in_channels
        self.gating_seq_len = gating_seq_len
        conv_gating         = []
        sigmoid             = []
        conv                = []
        relu                = []

        for i in range(len(self.in_channels)):
            conv_gating.append(nn.Conv2d(self.in_channels[i]*self.seq_len, self.seq_len, \
                                         kernel_size=3, padding=1))
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*self.seq_len, self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
        
        self.conv_gating = nn.ModuleList(conv_gating)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x):

        x_all = []
        x_temp = []

        for num_in in  range(len(self.in_channels)):
            x_all.append([x[seq][num_in] for seq in range(self.gating_seq_len)])

        for feat, conv_gating, sigmoid, conv, relu in zip(x_all, self.conv_gating, self.sigmoid, self.conv, self.relu):
            
            feat_cat      = torch.cat(feat, 1)
            gating_weight = sigmoid(conv_gating(feat_cat))
            feat_gating   = [feat[i] * gating_weight[:,i,:,:].unsqueeze(1) \
                             for i in range(len(feat))]

            feat_temp     = torch.cat(feat_gating, 1)
            feat_temp     = relu(conv(feat_temp))
            x_temp.append(feat_temp)

        return x_temp
