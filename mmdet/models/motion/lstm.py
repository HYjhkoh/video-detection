import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init
from spatial_correlation_sampler import spatial_correlation_sample


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