import torch
import torch.nn as nn
from spatial_correlation_sampler import spatial_correlation_sample

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init
from spatial_correlation_sampler import spatial_correlation_sample

import pdb
import numpy as np
import math
import logging
from mmcv.runner import load_checkpoint

@TEMPORAL.register_module
class ConvLSTM(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        x_convolution = []
        h_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        for i in range(len(self.in_channels)):
            x_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            h_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            
        self.x_convolution = nn.ModuleList(x_convolution)
        self.h_convolution = nn.ModuleList(h_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g    = nn.ModuleList(tanh_g)
        self.tanh_h    = nn.ModuleList(tanh_h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, x_pre, x):

        x_next = []
        for idx,(feat1,feat2,x_conv,h_conv,sigmoid_i,sigmoid_f,sigmoid_o,tanh_g,tanh_h) \
            in enumerate(zip(x_pre, x, self.x_convolution, self.h_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h)):
            h_pre, c_pre = feat1
            conv_x = x_conv(feat2) + h_conv(h_pre)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            x_next.append((h_next, c_next))

        return x_next

@TEMPORAL.register_module
class ConvLSTMReLU(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super(ConvLSTMReLU, self).__init__()
        self.in_channels = in_channels
        x_convolution = []
        h_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        relu_g = []
        relu_h = []
        for i in range(len(self.in_channels)):
            x_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            h_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            relu_g.append(nn.ReLU())
            relu_h.append(nn.ReLU())
            
        self.x_convolution = nn.ModuleList(x_convolution)
        self.h_convolution = nn.ModuleList(h_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.relu_g    = nn.ModuleList(relu_g)
        self.relu_h    = nn.ModuleList(relu_h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, x_pre, x):

        x_next = []
        for idx,(feat1,feat2,x_conv,h_conv,sigmoid_i,sigmoid_f,sigmoid_o,relu_g,relu_h) \
            in enumerate(zip(x_pre, x, self.x_convolution, self.h_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.relu_g, self.relu_h)):
            h_pre, c_pre = feat1
            conv_x = x_conv(feat2) + h_conv(h_pre)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = relu_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * relu_h(c_next)

            x_next.append((h_next, c_next))

        return x_next

@TEMPORAL.register_module
class ConvLSTMMotion(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size=3):
        super(ConvLSTMMotion, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        x_convolution = []
        h_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        for i in range(len(self.in_channels)):
            x_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            h_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            
        self.x_convolution = nn.ModuleList(x_convolution)
        self.h_convolution = nn.ModuleList(h_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g    = nn.ModuleList(tanh_g)
        self.tanh_h    = nn.ModuleList(tanh_h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, pre_states, x_pre, x):

        next_state = []
        for idx,(pre_state, feat1,feat2,x_conv,h_conv,sigmoid_i,sigmoid_f,sigmoid_o,tanh_g,tanh_h) in enumerate(zip(pre_states, x_pre, x, self.x_convolution, self.h_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h)):
            h_pre, c_pre = pre_state
            corr_feat = spatial_correlation_sample(feat2,
                                                   feat1,
                                                   kernel_size=1, 
                                                   patch_size=self.corr_size,
                                                   stride=1,
                                                   padding=0,
                                                   dilation_patch=1)
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w))
            corr_feat = corr_feat.view(b,1,ph,pw,h,w)
            pad_size = int((self.corr_size-1)/2)
            pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
            
            mul_feat = []
            for i in range(self.corr_size):
                for j in range(self.corr_size):
                    mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                    if i == 0 and j == 0:
                        new_feat = mul_feat
                    else:
                        new_feat += mul_feat
            
            conv_x = x_conv(new_feat) + h_conv(h_pre)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            next_state.append((h_next, c_next))

        return next_state

@TEMPORAL.register_module
class ConvLSTMPSLA(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size=3):
        super(ConvLSTMPSLA, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        update_conv1 = []
        update_conv2 = []
        update_conv3 = []
        update_sigmoid = []
        x_convolution = []
        h_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        for i in range(len(self.in_channels)):
            update_conv1.append(nn.Conv2d(in_channels[i]*2, 256,\
                                    kernel_size=1, padding=0))
            update_conv2.append(nn.Conv2d(256, 16,\
                                    kernel_size=3, padding=1))
            update_conv3.append(nn.Conv2d(16, 2,\
                                    kernel_size=1, padding=0))
            update_sigmoid.append(nn.Sigmoid())
            
            x_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            h_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            
        self.update_conv1 = nn.ModuleList(update_conv1)
        self.update_conv2 = nn.ModuleList(update_conv2)
        self.update_conv3 = nn.ModuleList(update_conv3)
        self.update_sigmoid = nn.ModuleList(update_sigmoid)

        self.x_convolution = nn.ModuleList(x_convolution)
        self.h_convolution = nn.ModuleList(h_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g    = nn.ModuleList(tanh_g)
        self.tanh_h    = nn.ModuleList(tanh_h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, pre_states, x_pre, x):

        next_state = []
        new_feats = []
        for idx,(pre_state, feat1,feat2,u_conv1, u_conv2, u_conv3, u_sigmoid,x_conv,h_conv,sigmoid_i,sigmoid_f,sigmoid_o,tanh_g,tanh_h) in enumerate(zip(pre_states, x_pre, x, self.update_conv1, self.update_conv2, self.update_conv3, self.update_sigmoid, self.x_convolution, self.h_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h)):
            h_pre, c_pre = pre_state
            corr_feat = spatial_correlation_sample(feat2,
                                                   feat1,
                                                   kernel_size=1, 
                                                   patch_size=self.corr_size,
                                                   stride=1,
                                                   padding=0,
                                                   dilation_patch=1)
            b, ph, pw, h, w = corr_feat.shape
            corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w))
            corr_feat = corr_feat.view(b,1,ph,pw,h,w)
            pad_size = int((self.corr_size-1)/2)
            pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
            
            mul_feat = []
            for i in range(self.corr_size):
                for j in range(self.corr_size):
                    mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                    if i == 0 and j == 0:
                        new_feat = mul_feat
                    else:
                        new_feat += mul_feat
            
            update_x = u_conv1(torch.cat([new_feat, feat2],1))
            update_x = u_conv2(update_x)
            update_weight = u_sigmoid(u_conv3(update_x))
            pdb.set_trace()
            update_weight = update_weight / (update_weight[:,0,:,:] + update_weight[:,1,:,:]).unsqueeze(1)

            update_feat = new_feat * update_weight[:,0,:,:].unsqueeze(1) + feat2 * update_weight[:,1,:,:].unsqueeze(1)
            new_feats.append(update_feat)
            pdb.set_trace()
            conv_x = x_conv(update_feat) + h_conv(h_pre)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)

            next_state.append((h_next, c_next))

        return next_state, new_feats

@TEMPORAL.register_module
class CorrLSTM(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None):
        super(CorrLSTM, self).__init__()
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

    def forward(self, x_pre, x, motion_state):

        state_next = []
        corr_feats = []
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
            corr_feats.append(corr_feat)

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

        return state_next, corr_feats

@TEMPORAL.register_module
class CorrLSTM_depth(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None):
        super(CorrLSTM_depth, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        depth_conv = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            depth_conv.append(nn.Conv2d(self.corr_size*self.corr_size, self.in_channels[i], kernel_size=3, padding=1))
            
        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.depth_conv = nn.ModuleList(depth_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)
                
    def forward(self, x_pre, x, motion_state):

        state_next = []
        corr_feats = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, depth_conv) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.depth_conv)):
        	
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
            corr_feats.append(corr_feat)

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
            # out = depth_conv(h_next)

            state_next.append((h_next, c_next))

        return state_next, corr_feats

@TEMPORAL.register_module
class CorrLSTM_resize(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None):
        super(CorrLSTM_resize, self).__init__()
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

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, x_pre, x, motion_state, corr_feat, re_idx):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h)):
        	
            # pdb.set_trace()
            if re_idx == 0:
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
                re_idx += 1
            else:
                if re_idx != 5:
                    corr_feat = self.avg_pool(corr_feat)
                    re_idx += 1
                else:
                    corr_feat = self.avg_pool1(corr_feat)

            # pdb.set_trace()
            # corr_feats.append(corr_feat)

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

        return state_next, corr_feat, re_idx

@TEMPORAL.register_module
class CorrLSTM_flow(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None):
        super(CorrLSTM_flow, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size

        leaky_relu = []
        relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        summary_conv = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d((corr_size*corr_size+int(in_channels[i]/8))*2, (corr_size*corr_size+int(in_channels[i]/8))*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            summary_conv.append(nn.Conv2d(in_channels[i], int(in_channels[i]/8), kernel_size=3, padding=1))
            relu.append(nn.ReLU())
            
        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.summary_conv = nn.ModuleList(summary_conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)
                
    def forward(self, x_pre, x, motion_state):

        state_next = []
        corr_feats = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, summary_conv, relu) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.summary_conv, self.relu)):
        	
            summary_feat = relu(summary_conv(feat1))

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
            corr_feats.append(corr_feat)

            new_feat = torch.cat([summary_feat, corr_feat], dim=1)

            h_pre, c_pre = state
            lstm_input = torch.cat([new_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, int(self.corr_size*self.corr_size+self.in_channels[idx]/8), dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = tanh_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * tanh_h(c_next)
            # out = depth_conv(h_next)

            state_next.append((h_next, c_next))

        return state_next, corr_feats

@TEMPORAL.register_module
class PSLAGFU(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size=3):
        super(PSLAGFU, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        conv_previous    = []
        sigmoid_previous = []
        conv_present     = []
        sigmoid_present  = []
        conv             = []
        relu             = []
        x_convolution = []
        h_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        relu_g = []
        relu_h = []
        for i in range(len(self.in_channels)):
            conv_previous.append(nn.Conv2d(in_channels[i]*2, 1, kernel_size=3, padding=1))
            sigmoid_previous.append(nn.Sigmoid())
            conv_present.append(nn.Conv2d(in_channels[i]*2,1,kernel_size=3,padding=1))
            sigmoid_present.append(nn.Sigmoid())

            conv.append(nn.Conv2d(in_channels[i]*2, in_channels[i], kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            
            x_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            h_convolution.append(nn.Conv2d(in_channels[i], in_channels[i]*4, \
                                    kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            relu_g.append(nn.ReLU(inplace=True))
            relu_h.append(nn.ReLU(inplace=True))
            
        self.conv_previous = nn.ModuleList(conv_previous)
        self.sigmoid_previous = nn.ModuleList(sigmoid_previous)
        self.conv_present = nn.ModuleList(conv_present)
        self.sigmoid_present = nn.ModuleList(sigmoid_present)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

        self.x_convolution = nn.ModuleList(x_convolution)
        self.h_convolution = nn.ModuleList(h_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.relu_g    = nn.ModuleList(relu_g)
        self.relu_h    = nn.ModuleList(relu_h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, pre_states, x_pre, x):

        next_state = []
        feats = []
        for idx,(pre_state, feat1,feat2,conv_previous, sigmoid_previous,conv_present,sigmoid_present,conv,relu,x_conv,h_conv,sigmoid_i,sigmoid_f,sigmoid_o,relu_g,relu_h) in enumerate(zip(pre_states, x_pre, x, self.conv_previous, self.sigmoid_previous, self.conv_present, self.sigmoid_present, self.conv, self.relu, self.x_convolution, self.h_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.relu_g, self.relu_h)):
            h_pre, c_pre = pre_state
            corr_feat = spatial_correlation_sample(feat2,
                                                   feat1,
                                                   kernel_size=1, 
                                                   patch_size=self.corr_size,
                                                   stride=1,
                                                   padding=0,
                                                   dilation_patch=1)
                                                   
            if corr_feat.shape[3] == 12 or corr_feat.shape[3] == 6:
                corr_feat /= 5

            b, ph, pw, h, w = corr_feat.shape
            # corr_feat_vis = corr_feat.view(b, ph*pw, h, w)
            # feats.append(corr_feat_vis)
            corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w))
            corr_feat = corr_feat.view(b,1,ph,pw,h,w)
            pad_size = int((self.corr_size-1)/2)
            pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
            
            for i in range(self.corr_size):
                for j in range(self.corr_size):
                    mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                    if i == 0 and j == 0:
                        new_feat = mul_feat
                    else:
                        new_feat += mul_feat
            
            feat_cat = torch.cat([new_feat, feat2], 1)
            previous_weight = sigmoid_previous(conv_previous(feat_cat))
            present_weight = sigmoid_present(conv_present(feat_cat))
            feat_previous = new_feat * previous_weight
            feat_present = feat2 * present_weight
            feat_gating = torch.cat([feat_previous, feat_present],1)
            feat_gating = relu(conv(feat_gating))

            conv_x = x_conv(feat_gating) + h_conv(h_pre)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = relu_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * relu_h(c_next)

            next_state.append((h_next, c_next))

            feats.append(feat_gating)

        return next_state, feats

@TEMPORAL.register_module
class CorrLSTM_upsample_avg(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_avg, self).__init__()
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
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            lstm_convolution.append(nn.Conv2d(self.corr_size*self.corr_size*2, self.corr_size*self.corr_size*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            tanh_g.append(nn.Tanh())
            tanh_h.append(nn.Tanh())
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))

        self.avg_pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.tanh_g = nn.ModuleList(tanh_g)
        self.tanh_h = nn.ModuleList(tanh_h)
        self.upsamples = nn.ModuleList(upsamples)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
                channel_num = int(len(m.bias)/4)
                nn.init.constant_(m.bias[channel_num:channel_num*2], 1)

    def forward(self, x_pre, x, motion_state):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples)):
        	
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
            for idx in range(int(math.log2(corr_feat.shape[3]/feat1.shape[3]))):
                if idx != 6:
                    corr_feat = self.avg_pooling1(corr_feat)
                else:
                    corr_feat = self.avg_pooling2(corr_feat)
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

@TEMPORAL.register_module
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

        self.avg_pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

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

    def forward(self, x_pre, x, motion_state):

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

@TEMPORAL.register_module
class CorrLSTM_upsample_resize_no(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_resize_no, self).__init__()
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

        self.avg_pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.last_downsample = nn.Upsample(scale_factor=1./3, mode='nearest')

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

    def forward(self, x_pre, x, motion_state):

        state_next = []
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, tanh_g, tanh_h, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.tanh_g, self.tanh_h, self.upsamples, self.downsamples)):
        	

            if feat1.shape[3] != 1:
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
            else:
                dw_corr_feat = self.last_downsample(dw_corr_feat)

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

@TEMPORAL.register_module
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


@TEMPORAL.register_module
class SubLSTM(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super(SubLSTM, self).__init__()
        self.in_channels = in_channels

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        tanh_g = []
        tanh_h = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            # import pdb;pdb.set_trace()
            lstm_convolution.append(nn.Conv2d(self.in_channels[i]*2, self.in_channels[i]*4, kernel_size=3, padding=1))
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

@TEMPORAL.register_module
class CorrLSTM_upsample_resize_norm_relu(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(CorrLSTM_upsample_resize_norm_relu, self).__init__()
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
            dw_corr_feat = dw_corr_feat / (dw_corr_feat).max()

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

@TEMPORAL.register_module
class SubLSTM_relu(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super(SubLSTM_relu, self).__init__()
        self.in_channels = in_channels

        leaky_relu = []
        lstm_convolution = []
        sigmoid_i = []
        sigmoid_f = []
        sigmoid_o = []
        relu_g = []
        relu_h = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            # import pdb;pdb.set_trace()
            lstm_convolution.append(nn.Conv2d(self.in_channels[i]*2, self.in_channels[i]*4, kernel_size=3, padding=1))
            sigmoid_i.append(nn.Sigmoid())
            sigmoid_f.append(nn.Sigmoid())
            sigmoid_o.append(nn.Sigmoid())
            relu_g.append(nn.ReLU())
            relu_h.append(nn.ReLU())

        self.leaky_relu = nn.ModuleList(leaky_relu)
        self.lstm_convolution = nn.ModuleList(lstm_convolution)
        self.sigmoid_i = nn.ModuleList(sigmoid_i)
        self.sigmoid_f = nn.ModuleList(sigmoid_f)
        self.sigmoid_o = nn.ModuleList(sigmoid_o)
        self.relu_g = nn.ModuleList(relu_g)
        self.relu_h = nn.ModuleList(relu_h)

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
        for idx, (feat1, feat2, state, leaky_relu, lstm_conv, sigmoid_i, sigmoid_f, sigmoid_o, relu_g, relu_h) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.lstm_convolution, self.sigmoid_i, self.sigmoid_f, self.sigmoid_o, self.relu_g, self.relu_h)):

            sub_feat = feat1 - feat2

            h_pre, c_pre = state
            lstm_input = torch.cat([sub_feat, h_pre], dim=1)
            conv_x = lstm_conv(lstm_input)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_x, self.in_channels[idx], dim=1)
            i = sigmoid_i(cc_i)
            f = sigmoid_f(cc_f)
            o = sigmoid_o(cc_o)
            g = relu_g(cc_g)
            c_next = f * c_pre + i * g
            h_next = o * relu_h(c_next)

            state_next.append((h_next, c_next))

        return state_next

@TEMPORAL.register_module
class Corr_upsample_resize_norm(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size = None,
                 upsample_param = None):
        super(Corr_upsample_resize_norm, self).__init__()
        self.in_channels = in_channels
        self.corr_size = corr_size
        self.upsample_param = upsample_param

        leaky_relu = []
        upsamples = []
        downsamples = []
        for i in range(len(self.in_channels)):
            leaky_relu.append(nn.LeakyReLU(inplace=True))
            upsamples.append(nn.Upsample(scale_factor=upsample_param*(pow(2, i)), mode='nearest'))
            downsamples.append(nn.Upsample(scale_factor=1./pow(2,i)/upsample_param, mode='nearest'))

        self.leaky_relu = nn.ModuleList(leaky_relu)
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
        for idx, (feat1, feat2, state, leaky_relu, upsamples, downsamples) in enumerate(zip(x, x_pre, motion_state, self.leaky_relu, self.upsamples, self.downsamples)):
        	
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

            state_next.append((dw_corr_feat, corr_feat))

        return state_next

@TEMPORAL.register_module
class PixelGatingAlign(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size=5,
                 gating_seq_len=4):
        super(PixelGatingAlign, self).__init__()
        self.in_channels     = in_channels
        self.corr_size       = corr_size
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        conv_down            = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_gating.append(nn.Conv2d(self.in_channels[i]*2, 
                                             2, \
                                             kernel_size=3, padding=1))
                conv_down.append(nn.Conv2d(self.in_channels[i]*2, 
                                             self.in_channels[i], \
                                             kernel_size=3, padding=1))                    

        self.conv_gating = nn.ModuleList(conv_gating)
        self.conv_down = nn.ModuleList(conv_down)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)

        else:
            if 'gating' in pretrained:
                # import pdb; pdb.set_trace()
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)

            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x):

        feats = []
        x_temp = []
        k = 0

        for num_in in  range(len(self.in_channels)):
            for seq in range(self.gating_seq_len-1):
                feats.append([x[self.gating_seq_len-1][num_in],x[seq][num_in]])

        for sigmoid, conv, relu in zip(self.sigmoid, self.conv, self.relu):
            
            feat_gated_all = []

            channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
            for seq in range(self.gating_seq_len-1):
                [feat2, feat1] = feats[k*(self.gating_seq_len-1)+seq]
                corr_feat = spatial_correlation_sample(feat2,
                                                       feat1,
                                                       kernel_size=1, 
                                                       patch_size=self.corr_size,
                                                       stride=1,
                                                       padding=0,
                                                       dilation_patch=1)

                b, ph, pw, h, w = corr_feat.shape
                corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w)*20/(corr_feat.max()))
                corr_feat = corr_feat.view(b,1,ph,pw,h,w)
                pad_size = int((self.corr_size-1)/2)
                pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
                
                for i in range(self.corr_size):
                    for j in range(self.corr_size):
                        mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                        if i == 0 and j == 0:
                            new_feat = mul_feat
                        else:
                            new_feat += mul_feat
                feats[k*(self.gating_seq_len-1)+seq][-1] = new_feat
                
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = [feat[:,i*channel:(i+1)*channel,:,:] * gating_weight[:,i,:,:].unsqueeze(1) \
                            for i in range(2)]
                feat_gated = torch.cat(feat_gated, 1)
                feat_gated = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated))
                feat_gated_all.append(feat_gated)

            feat_total = torch.cat(feat_gated_all, 1)
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

        return x_temp

@TEMPORAL.register_module
class PixelGatingAlignFuture(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size=5,
                 gating_seq_len=5):
        super(PixelGatingAlignFuture, self).__init__()
        self.in_channels     = in_channels
        self.corr_size       = corr_size
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        conv_down            = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_gating.append(nn.Conv2d(self.in_channels[i]*2, 
                                             2, \
                                             kernel_size=3, padding=1))
                conv_down.append(nn.Conv2d(self.in_channels[i]*2, 
                                             self.in_channels[i], \
                                             kernel_size=3, padding=1))                    

        self.conv_gating = nn.ModuleList(conv_gating)
        self.conv_down = nn.ModuleList(conv_down)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)

        else:
            if 'gating' in pretrained:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)

            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x):

        feats = []
        x_temp = []
        k = 0

        # import pdb; pdb.set_trace()
        for num_in in  range(len(self.in_channels)):
            for seq in range(self.gating_seq_len):
                if seq == int((self.gating_seq_len-1)/2):
                    continue
                else:
                    feats.append([x[int((self.gating_seq_len-1)/2)][num_in],x[seq][num_in]])

        for sigmoid, conv, relu in zip(self.sigmoid, self.conv, self.relu):
            
            feat_gated_all = []

            channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
            for seq in range(self.gating_seq_len-1):
                [feat2, feat1] = feats[k*(self.gating_seq_len-1)+seq]
                corr_feat = spatial_correlation_sample(feat2,
                                                       feat1,
                                                       kernel_size=1, 
                                                       patch_size=self.corr_size,
                                                       stride=1,
                                                       padding=0,
                                                       dilation_patch=1)

                b, ph, pw, h, w = corr_feat.shape
                corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w)*20/(corr_feat.max()))
                corr_feat = corr_feat.view(b,1,ph,pw,h,w)
                pad_size = int((self.corr_size-1)/2)
                pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
                
                for i in range(self.corr_size):
                    for j in range(self.corr_size):
                        mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                        if i == 0 and j == 0:
                            new_feat = mul_feat
                        else:
                            new_feat += mul_feat
                feats[k*(self.gating_seq_len-1)+seq][-1] = new_feat
                
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = [feat[:,i*channel:(i+1)*channel,:,:] * gating_weight[:,i,:,:].unsqueeze(1) \
                            for i in range(2)]
                feat_gated = torch.cat(feat_gated, 1)
                feat_gated = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated))
                feat_gated_all.append(feat_gated)

            feat_total = torch.cat(feat_gated_all, 1)
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

        return x_temp

@TEMPORAL.register_module
class PixelGatingAlignFutureRoi(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 corr_size=5,
                 gating_seq_len=5):
        super(PixelGatingAlignFutureRoi, self).__init__()
        self.in_channels     = in_channels
        self.corr_size       = corr_size
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        conv_down            = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(1):
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_gating.append(nn.Conv2d(self.in_channels[i]*2, 
                                             2, \
                                             kernel_size=3, padding=1))
                conv_down.append(nn.Conv2d(self.in_channels[i]*2, 
                                             self.in_channels[i], \
                                             kernel_size=3, padding=1))                    

        self.conv_gating = nn.ModuleList(conv_gating)
        self.conv_down = nn.ModuleList(conv_down)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)

        else:
            if 'gating' in pretrained:
                # import pdb; pdb.set_trace()
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)

            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x):

        feats = []
        x_temp = []
        k = 0

        for seq in range(self.gating_seq_len):
            if seq == int((self.gating_seq_len-1)/2):
                continue
            else:
                feats.append([x[int((self.gating_seq_len-1)/2)],x[seq]])

        for sigmoid, conv, relu in zip(self.sigmoid, self.conv, self.relu):
            
            feat_gated_all = []

            channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
            for seq in range(self.gating_seq_len-1):
                [feat2, feat1] = feats[k*(self.gating_seq_len-1)+seq]
                corr_feat = spatial_correlation_sample(feat2,
                                                       feat1,
                                                       kernel_size=1, 
                                                       patch_size=self.corr_size,
                                                       stride=1,
                                                       padding=0,
                                                       dilation_patch=1)

                b, ph, pw, h, w = corr_feat.shape
                corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w)*20/(corr_feat.max()))
                corr_feat = corr_feat.view(b,1,ph,pw,h,w)
                pad_size = int((self.corr_size-1)/2)
                pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
                
                for i in range(self.corr_size):
                    for j in range(self.corr_size):
                        mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
                        if i == 0 and j == 0:
                            new_feat = mul_feat
                        else:
                            new_feat += mul_feat
                feats[k*(self.gating_seq_len-1)+seq][-1] = new_feat
                
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = [feat[:,i*channel:(i+1)*channel,:,:] * gating_weight[:,i,:,:].unsqueeze(1) \
                            for i in range(2)]
                feat_gated = torch.cat(feat_gated, 1)
                feat_gated = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated))
                feat_gated_all.append(feat_gated)

            feat_total = torch.cat(feat_gated_all, 1)
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

        return x_temp

@TEMPORAL.register_module
class PixelGatingonce_three2_future_roi(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce_three2_future_roi, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        conv_down            = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(1):
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_gating.append(nn.Conv2d(self.in_channels[i]*2, 
                                             2, \
                                             kernel_size=3, padding=1))
                conv_down.append(nn.Conv2d(self.in_channels[i]*2, 
                                             self.in_channels[i], \
                                             kernel_size=3, padding=1))                    

        self.conv_gating = nn.ModuleList(conv_gating)
        self.conv_down = nn.ModuleList(conv_down)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)

    def init_weights(self, pretrained):
        if pretrained == None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform', bias=0)

        else:
            if 'gating' in pretrained:
                # import pdb; pdb.set_trace()
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)

            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x):

        feats = []
        x_temp = []
        k = 0
        for seq in range(self.gating_seq_len):
                if seq == int((self.gating_seq_len-1)/2):
                    continue
                else:
                    feats.append([x[int((self.gating_seq_len-1)/2)],x[seq]])
        
        for sigmoid, conv, relu in zip(self.sigmoid, self.conv, self.relu):
            
            feat_gated_all = []
            channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
            for seq in range(self.gating_seq_len-1):
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = [feat[:,i*channel:(i+1)*channel,:,:] * gating_weight[:,i,:,:].unsqueeze(1) \
                            for i in range(2)]
                feat_gated = torch.cat(feat_gated, 1)
                feat_gated = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated))
                feat_gated_all.append(feat_gated)

            feat_total = torch.cat(feat_gated_all, 1)
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

        return x_temp