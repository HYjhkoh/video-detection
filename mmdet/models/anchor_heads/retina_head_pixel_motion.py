import logging

import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from .anchor_head_corr import AnchorHeadCorr
from ..registry import HEADS
from mmcv.runner import load_checkpoint
from ..utils import bias_init_with_prob, ConvModule

import pdb

@HEADS.register_module
class RetinaHeadPixelMotion(AnchorHeadCorr):

    def __init__(self,
                 num_classes,
                 corr_size,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.corr_size = corr_size
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHeadPixelMotion, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        # self.red_convs = nn.Conv2d(self.in_channels,
        #                            self.in_channels,
        #                            kernel_size=3,
        #                            stride=1,
        #                            padding=1)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)


        self.depth_states = nn.Conv2d(self.corr_size*self.corr_size, self.in_channels, kernel_size=1, padding=0)
        self.depth_convs = nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=3, padding=1)

    def init_weights(self, pretrained):
        if pretrained == 'pretrained/retina_vid.pth':
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained == 'pretrained/retina_det.pth':
            weight = torch.load(pretrained)
            for idx,m in enumerate(self.cls_convs):
                m.conv.weight = torch.nn.Parameter(weight['cls_convs.%d.conv.weight'%idx])
                m.conv.bias = torch.nn.Parameter(weight['cls_convs.%d.conv.bias'%idx])
            print('classification weight Success')
            for idx,m in enumerate(self.reg_convs):
                m.conv.weight = torch.nn.Parameter(weight['reg_convs.%d.conv.weight'%idx])
                m.conv.bias = torch.nn.Parameter(weight['reg_convs.%d.conv.bias'%idx])
            print('regression weight Success')
            
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.retina_cls, std=0.01, bias=bias_cls)
            normal_init(self.retina_reg, std=0.01)
        else:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.retina_cls, std=0.01, bias=bias_cls)
            normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x_pixel, x_motion):
        # pdb.set_trace()
        x_motion = self.relu(self.depth_states(x_motion))
        feat = torch.cat([x_pixel, x_motion], 1)
        feat = self.relu(self.depth_convs(feat))
        # cls_feat = self.relu(self.red_convs(feat))
        cls_feat = feat
        reg_feat = cls_feat
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
