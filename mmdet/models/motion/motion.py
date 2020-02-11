import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import MOTION
from mmcv.cnn import xavier_init
from spatial_correlation_sampler import spatial_correlation_sample
from mmcv.runner import load_checkpoint

import pdb
import logging

@MOTION.register_module
class MotionCorrLSTM(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None):
        super(MotionCorrLSTM, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            
        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h)):
            
            corr_feat = spatial_correlation_sample(feat1,
                                                   feat2,
                                                   kernel_size=1,
                                                   patch_size=self.corr_size,
                                                   stride=1,
                                                   padding=0,
                                                   dilation_patch=1)

            # if corr_feat.shape[3] == 12 or corr_feat.shape[3] == 6:
            #     corr_feat /= 5

            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat)

            h_pre, c_pre = state
            lstm_input = torch.cat([corr_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.corr_size*self.corr_size, dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@MOTION.register_module
class CorrLSTM_upsample_resize(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_resize, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        self.upsample_param = upsample_param

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.upsamples = nn.ModuleList(upsamples)
        self.downsamples = nn.ModuleList(downsamples)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)
                    channel_num = int(len(m.bias)/4)
                    nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

        else:
            if 'corr' in pretrained:
                # import pdb; pdb.set_trace()
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                        channel_num = int(len(m.bias)/4)
                        nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples, self.downsamples)):
        	
            up_feat1 = upsamples(feat1)
            up_feat2 = upsamples(feat2)
            corr_feat = spatial_correlation_sample(up_feat1,
                                                up_feat2,
                                                kernel_size=1,
                                                patch_size=self.corr_size,
                                                stride=1,
                                                padding=0,
                                                dilation_patch=1)            
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat)
            dw_corr_feat = downsamples(corr_feat)
            # pdb.set_trace()
            h_pre, c_pre = state
            lstm_input = torch.cat([dw_corr_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.corr_size*self.corr_size, dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@MOTION.register_module
class CorrLSTM_upsample_resize_PSLA(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_resize_PSLA, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        self.upsample_param = upsample_param

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.upsamples = nn.ModuleList(upsamples)
        self.downsamples = nn.ModuleList(downsamples)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        feats = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples, self.downsamples)):
        	
            up_feat1 = upsamples(feat1)
            up_feat2 = upsamples(feat2)
            corr_feat = spatial_correlation_sample(up_feat1,
                                                up_feat2,
                                                kernel_size=1,
                                                patch_size=self.corr_size,
                                                stride=1,
                                                padding=0,
                                                dilation_patch=1)       
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat)
            dw_corr_feat = downsamples(corr_feat)

            b, pa, h, w = dw_corr_feat.shape
            # corr_feat_vis = corr_feat.view(b, ph*pw, h, w)
            # feats.append(corr_feat_vis)
            dw_corr_feat_p = F.softmax(dw_corr_feat.view(b,ph*pw,h,w))
            dw_corr_feat_p = dw_corr_feat_p.view(b,1,ph,pw,h,w)
            pad_size = int((self.corr_size-1)/2)
            pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
            
            for i in range(self.corr_size):
                for j in range(self.corr_size):
                    mul_feat = dw_corr_feat_p[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                    if i == 0 and j == 0:
                        new_feat = mul_feat
                    else:
                        new_feat += mul_feat
            feats.append(new_feat)

            # pdb.set_trace()
            h_pre, c_pre = state
            lstm_input = torch.cat([dw_corr_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.corr_size*self.corr_size, dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next, feats
        

@MOTION.register_module
class CorrLSTM_upsample_resize_relu(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_resize_relu, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        self.upsample_param = upsample_param

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        relu_g = []
        relu_h = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            relu_g.append(nn.ReLU())
            relu_h.append(nn.ReLU())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.relu_g = nn.ModuleList(relu_g)
        self.relu_h = nn.ModuleList(relu_h)
        self.upsamples = nn.ModuleList(upsamples)
        self.downsamples = nn.ModuleList(downsamples)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)
                    channel_num = int(len(m.bias)/4)
                    nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

        else:
            if 'corr' in pretrained:
                # import pdb; pdb.set_trace()
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                        channel_num = int(len(m.bias)/4)
                        nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, relu_g, relu_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.relu_g, self.relu_h, self.upsamples, self.downsamples)):
        	
            up_feat1 = upsamples(feat1)
            up_feat2 = upsamples(feat2)
            corr_feat = spatial_correlation_sample(up_feat1,
                                                up_feat2,
                                                kernel_size=1,
                                                patch_size=self.corr_size,
                                                stride=1,
                                                padding=0,
                                                dilation_patch=1)            
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat)
            dw_corr_feat = downsamples(corr_feat)
            # pdb.set_trace()
            h_pre, c_pre = state
            lstm_input = torch.cat([dw_corr_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.corr_size*self.corr_size, dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = relu_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * relu_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@MOTION.register_module
class CorrLSTM_upsample_resize_norm(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_resize_norm, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        self.upsample_param = upsample_param

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.upsamples = nn.ModuleList(upsamples)
        self.downsamples = nn.ModuleList(downsamples)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)
                    channel_num = int(len(m.bias)/4)
                    nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

        else:
            if 'corr' in pretrained:
                # import pdb; pdb.set_trace()
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                        channel_num = int(len(m.bias)/4)
                        nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples, self.downsamples)):
        	
            up_feat1 = upsamples(feat1)
            up_feat2 = upsamples(feat2)
            corr_feat = spatial_correlation_sample(up_feat1,
                                                up_feat2,
                                                kernel_size=1,
                                                patch_size=self.corr_size,
                                                stride=1,
                                                padding=0,
                                                dilation_patch=1)            
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat)
            dw_corr_feat = downsamples(corr_feat)
            dw_corr_feat = dw_corr_feat / (dw_corr_feat).max()

            h_pre, c_pre = state
            lstm_input = torch.cat([dw_corr_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.corr_size*self.corr_size, dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@MOTION.register_module
class SubLSTM_upsample_resize_norm(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 upsample_param = None):
        super(SubLSTM_upsample_resize_norm, self).__init__()
        self.in_channels = in_channels
        self.upsample_param = upsample_param

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            # import pdb;pdb.set_trace()
            lstm_convolution.append(nn.Conv2d(self.in_channels[i]*2, self.in_channels[i]*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.upsamples = nn.ModuleList(upsamples)
        self.downsamples = nn.ModuleList(downsamples)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)
                    channel_num = int(len(m.bias)/4)
                    nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

        else:
            if 'corr' in pretrained:
                # import pdb; pdb.set_trace()
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                        channel_num = int(len(m.bias)/4)
                        nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples, self.downsamples)):
            
            up_feat1 = upsamples(feat1)
            up_feat2 = upsamples(feat2)
            sub_feat = up_feat1 - up_feat2
            dw_sub_feat = downsamples(sub_feat)

            # dw_sub_feat = dw_sub_feat / (dw_sub_feat).max()

            h_pre, c_pre = state
            lstm_input = torch.cat([dw_sub_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@MOTION.register_module
class SubLSTM(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 upsample_param = None):
        super(SubLSTM, self).__init__()
        self.in_channels = in_channels
        self.upsample_param = upsample_param

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            # import pdb;pdb.set_trace()
            lstm_convolution.append(nn.Conv2d(self.in_channels[i]*2, self.in_channels[i]*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.upsamples = nn.ModuleList(upsamples)
        self.downsamples = nn.ModuleList(downsamples)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)
                    channel_num = int(len(m.bias)/4)
                    nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

        else:
            if 'corr' in pretrained:
                # import pdb; pdb.set_trace()
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                        channel_num = int(len(m.bias)/4)
                        nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples, self.downsamples)):

            sub_feat = feat1 - feat2

            h_pre, c_pre = state
            lstm_input = torch.cat([sub_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@MOTION.register_module
class CorrLSTM_resize_norm(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None):
        super(CorrLSTM_resize_norm, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)
                    channel_num = int(len(m.bias)/4)
                    nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

        else:
            if 'corr' in pretrained:
                # import pdb; pdb.set_trace()
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                        channel_num = int(len(m.bias)/4)
                        nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, motion_state, x_pre, x):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h)):
        	
            corr_feat = spatial_correlation_sample(feat1,
                                                feat2,
                                                kernel_size=1,
                                                patch_size=self.corr_size,
                                                stride=1,
                                                padding=0,
                                                dilation_patch=1)            
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = corr_feat.view(b, ph*pw, h, w)
            corr_feat = leaky_relu(corr_feat)
            dw_corr_feat = corr_feat / (corr_feat).max()

            h_pre, c_pre = state
            lstm_input = torch.cat([dw_corr_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.corr_size*self.corr_size, dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            state_next.append((h_next, c_next))

        return state_next