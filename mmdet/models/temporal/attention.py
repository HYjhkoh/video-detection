import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init
from .concat import Concat

@TEMPORAL.register_module
class Attention(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 feature_size=(38, 19, 10, 5, 3, 1),
                 reduction_ratio=16):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        conv1 = []
        relu1 = []
        conv2 = []
        relu2 = []

        channel_avgpool = []
        channel_maxpool = []
        channel_conv1 = []
        channel_conv2 = []
        channel_sigmoid = []

        spatial_avgpool = []
        spatial_maxpool = []
        spatial_conv = []
        spatial_sigmoid = []

        for i in range(len(self.in_channels)):
            conv1.append(nn.Conv2d(in_channels[i]*2,in_channels[i],kernel_size=3,padding=1))
            relu1.append(nn.ReLU(inplace=True))
            conv2.append(nn.Conv2d(in_channels[i],in_channels[i],kernel_size=3,padding=1))
            relu2.append(nn.ReLU(inplace=True))

            channel_avgpool.append(nn.AvgPool2d(kernel_size=self.feature_size[i], stride=1))
            channel_maxpool.append(nn.MaxPool2d(kernel_size=self.feature_size[i], stride=1))
            channel_conv1.append(
                nn.Conv2d(in_channels[i],int(in_channels[i]/reduction_ratio),\
                          kernel_size=3,padding=1))
            channel_conv2.append(
                nn.Conv2d(int(in_channels[i]/reduction_ratio), in_channels[i],\
                          kernel_size=3,padding=1))
            channel_sigmoid.append(nn.Sigmoid())

            spatial_avgpool.append(ChannelAvgPool(kernel_size=self.in_channels[i]))
            spatial_maxpool.append(ChannelMaxPool(kernel_size=self.in_channels[i]))
            spatial_conv.append(nn.Conv2d(2, 1, kernel_size=7, padding=3))
            spatial_sigmoid.append(nn.Sigmoid())

        self.conv1 = nn.ModuleList(conv1)
        self.relu1 = nn.ModuleList(relu1)
        self.conv2 = nn.ModuleList(conv2)
        self.relu2 = nn.ModuleList(relu2)

        self.channel_avgpool = nn.ModuleList(channel_avgpool)
        self.channel_maxpool = nn.ModuleList(channel_maxpool)
        self.channel_conv1 = nn.ModuleList(channel_conv1)
        self.channel_conv2 = nn.ModuleList(channel_conv2)
        self.channel_sigmoid = nn.ModuleList(channel_sigmoid)

        self.spatial_avgpool = nn.ModuleList(spatial_avgpool)
        self.spatial_maxpool = nn.ModuleList(spatial_maxpool)
        self.spatial_conv = nn.ModuleList(spatial_conv)
        self.spatial_sigmoid = nn.ModuleList(spatial_sigmoid)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x_pre, x):

        x_temp = []
        for feat1, feat2, conv1, relu1, conv2, relu2, channel_avgpool, channel_maxpool, channel_conv1, channel_conv2, channel_sigmoid, spatial_avgpool, spatial_maxpool, spatial_conv, spatial_sigmoid in zip(x_pre, x, self.conv1, self.relu1, self.conv2, self.relu2, self.channel_avgpool, self.channel_maxpool, self.channel_conv1, self.channel_conv2, self.channel_sigmoid, self.spatial_avgpool, self.spatial_maxpool, self.spatial_conv, self.spatial_sigmoid):

            feat_concat = torch.cat([feat1, feat2], 1)
            feat_concat = relu1(conv1(feat_concat))
            feat_concat = relu2(conv2(feat_concat))

            channel_avg_attention = channel_avgpool(feat_concat)
            channel_max_attention = channel_maxpool(feat_concat)
            channel_avg_attention = channel_conv2(channel_conv1(channel_avg_attention))
            channel_max_attention = channel_conv2(channel_conv1(channel_max_attention))
            channel_attention = channel_sigmoid(channel_avg_attention + channel_max_attention)
            feat_channel_attention = feat_concat * channel_attention

            spatial_avg_attention = spatial_avgpool(feat_channel_attention)
            spatial_max_attention = spatial_maxpool(feat_channel_attention)
            spatial_attention = spatial_conv(torch.cat([spatial_avg_attention,\
                                                       spatial_max_attention], 1))
            spatial_attention = spatial_sigmoid(spatial_attention)

            attention = spatial_attention * feat_channel_attention
            x_temp.append(feat_concat + attention)


        return x_temp

class ChannelAvgPool(nn.AvgPool1d):
    def forward(self, input):
        n, c, h, w = input.size()
        input = input.view(n,c,h*w).permute(0,2,1)
        pooled =  F.avg_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.ceil_mode,self.count_include_pad)
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,1,h,w)

class ChannelMaxPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, h, w = input.size()
        input = input.view(n,c,h*w).permute(0,2,1)
        pooled =  F.max_pool1d(input, self.kernel_size, self.stride, self.padding)
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,1,h,w)
