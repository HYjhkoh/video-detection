import torch
import torch.nn as nn

from .. import builder
from ..registry import PIXEL
from mmcv.cnn import xavier_init
from mmcv.runner import load_checkpoint

import pdb
import logging

@PIXEL.register_module
class PixelGatingSeq(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingSeq, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            conv_gating.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.gating_seq_len, \
                                         kernel_size=3, padding=1))
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
        
        self.conv_gating = nn.ModuleList(conv_gating)
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

@PIXEL.register_module
class PixelGatingonce(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            conv_gating.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, 
                                         self.in_channels[i]*self.gating_seq_len, \
                                         kernel_size=3, padding=1))
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
        
        self.conv_gating = nn.ModuleList(conv_gating)
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

        x_all = []
        x_temp = []

        for num_in in  range(len(self.in_channels)):
            x_all.append([x[seq][num_in] for seq in range(self.gating_seq_len)])

        for feat, conv_gating, sigmoid, conv, relu in zip(x_all, self.conv_gating, self.sigmoid, self.conv, self.relu):
            
            feat_cat      = torch.cat(feat, 1)
            gating_weight = sigmoid(conv_gating(feat_cat))
            # channel = int(gating_weight.shape[1] / 4)

            # feat_gating   = [feat[i] * gating_weight[:,i*channel:(i+1)*channel,:,:] \
            #                  for i in range(len(feat))]
            feat_gating = feat_cat * gating_weight

            # feat_temp     = torch.cat(feat_gating, 1)
            feat_temp     = relu(conv(feat_gating))
            x_temp.append(feat_temp)
        return x_temp

@PIXEL.register_module
class PixelGatingseperate(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingseperate, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        self.feat_size = [38, 19, 10, 5, 3, 1]
        conv_gating_channel_max  = []
        conv_gating_channel_avg  = []
        conv_gating_pixel    = []
        sigmoid              = []
        conv                 = []
        relu                 = []
        avg_pooling          = []
        max_pooling          = []

        for i in range(len(self.in_channels)):
            conv_gating_pixel.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.gating_seq_len, \
                                         kernel_size=3, padding=1))
            sigmoid.append(nn.Sigmoid())
            conv_gating_channel_max.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i]*self.gating_seq_len, \
                                         kernel_size=1, padding=0))
            conv_gating_channel_avg.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i]*self.gating_seq_len, \
                                         kernel_size=1, padding=0))
            conv.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            avg_pooling.append(nn.AvgPool2d(kernel_size=self.feat_size[i], stride=1, padding=0))
            max_pooling.append(nn.MaxPool2d(kernel_size=self.feat_size[i], stride=1, padding=0))
        
        self.conv_gating_pixel = nn.ModuleList(conv_gating_pixel)
        self.conv_gating_channel_max = nn.ModuleList(conv_gating_channel_max)
        self.conv_gating_channel_avg = nn.ModuleList(conv_gating_channel_avg)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)
        self.avg_pooling = nn.ModuleList(avg_pooling)
        self.max_pooling = nn.ModuleList(max_pooling)

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

        x_all = []
        x_temp = []

        for num_in in  range(len(self.in_channels)):
            x_all.append([x[seq][num_in] for seq in range(self.gating_seq_len)])

        for feat, conv_gating_pixel, conv_gating_channel_max, conv_gating_channel_avg, sigmoid, conv, relu, avg_pooling, max_pooling in zip(x_all, self.conv_gating_pixel, self.conv_gating_channel_max, self.conv_gating_channel_avg, self.sigmoid, self.conv, self.relu, self.avg_pooling, self.max_pooling):
            
            feat_cat      = torch.cat(feat, 1)
            avg_pooled_weight = (conv_gating_channel_avg(avg_pooling(feat_cat)))
            max_pooled_weight = (conv_gating_channel_max(max_pooling(feat_cat)))
            channel_weight = sigmoid(avg_pooled_weight + max_pooled_weight)
            feat_channel  = feat_cat * channel_weight

            pixel_weight = sigmoid(conv_gating_pixel(feat_channel))
            channel = int(feat_channel.shape[1] / 4)
            feat_gating   = [feat_channel[:,i*channel:(i+1)*channel,:,:] * pixel_weight[:,i,:,:].unsqueeze(1) \
                             for i in range(len(feat))]

            feat_temp     = torch.cat(feat_gating, 1)
            feat_temp     = relu(conv(feat_temp))
            x_temp.append(feat_temp)
        return x_temp

@PIXEL.register_module
class PixelGatingseperate_all(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingseperate_all, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        self.feat_size = [38, 19, 10, 5, 3, 1]
        conv_gating_channel_max  = []
        conv_gating_channel_avg  = []
        conv_gating_pixel    = []
        sigmoid              = []
        conv                 = []
        relu                 = []
        avg_pooling          = []
        max_pooling          = []

        for i in range(len(self.in_channels)):
            conv_gating_pixel.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.gating_seq_len, \
                                         kernel_size=3, padding=1))
            sigmoid.append(nn.Sigmoid())
            conv_gating_channel_max.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i]*self.gating_seq_len, \
                                         kernel_size=1, padding=0))
            conv_gating_channel_avg.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i]*self.gating_seq_len, \
                                         kernel_size=1, padding=0))
            conv.append(nn.Conv2d(self.in_channels[i]*self.gating_seq_len, self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            avg_pooling.append(nn.AvgPool2d(kernel_size=self.feat_size[i], stride=1, padding=0))
            max_pooling.append(nn.MaxPool2d(kernel_size=self.feat_size[i], stride=1, padding=0))
        
        self.conv_gating_pixel = nn.ModuleList(conv_gating_pixel)
        self.conv_gating_channel_max = nn.ModuleList(conv_gating_channel_max)
        self.conv_gating_channel_avg = nn.ModuleList(conv_gating_channel_avg)
        self.sigmoid = nn.ModuleList(sigmoid)
        self.conv = nn.ModuleList(conv)
        self.relu = nn.ModuleList(relu)
        self.avg_pooling = nn.ModuleList(avg_pooling)
        self.max_pooling = nn.ModuleList(max_pooling)

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

        x_all = []
        x_temp = []

        for num_in in  range(len(self.in_channels)):
            x_all.append([x[seq][num_in] for seq in range(self.gating_seq_len)])

        for feat, conv_gating_pixel, conv_gating_channel_max, conv_gating_channel_avg, sigmoid, conv, relu, avg_pooling, max_pooling in zip(x_all, self.conv_gating_pixel, self.conv_gating_channel_max, self.conv_gating_channel_avg, self.sigmoid, self.conv, self.relu, self.avg_pooling, self.max_pooling):
            
            feat_cat      = torch.cat(feat, 1)
            avg_pooled_weight = (conv_gating_channel_avg(avg_pooling(feat_cat)))
            max_pooled_weight = (conv_gating_channel_max(max_pooling(feat_cat)))
            channel_weight = sigmoid(avg_pooled_weight + max_pooled_weight)
            feat_channel  = feat_cat * channel_weight

            pixel_weight = sigmoid(conv_gating_pixel(feat_cat))
            channel = feat[0].shape[1]
            feat_gating   = [feat_channel[:,i*channel:(i+1)*channel,:,:] * pixel_weight[:,i,:,:].unsqueeze(1) \
                             for i in range(len(feat))]

            feat_temp     = torch.cat(feat_gating, 1)
            feat_temp     = relu(conv(feat_temp))
            x_temp.append(feat_temp)
        return x_temp

@PIXEL.register_module
class PixelGatingonce_three(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce_three, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        # conv_gating2         = []
        # conv_gating3         = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            # conv_gating1.append(nn.Conv2d(self.in_channels[i]*2, 
            #                              self.in_channels[i], \
            #                              kernel_size=3, padding=1))
            # conv_gating2.append(nn.Conv2d(self.in_channels[i]*2, 
            #                              self.in_channels[i], \
            #                              kernel_size=3, padding=1))
            # conv_gating3.append(nn.Conv2d(self.in_channels[i]*2, 
            #                              self.in_channels[i], \
            #                              kernel_size=3, padding=1))            
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_gating.append(nn.Conv2d(self.in_channels[i]*2, 
                                             self.in_channels[i], \
                                             kernel_size=3, padding=1))

        self.conv_gating = nn.ModuleList(conv_gating)
        # self.conv_gating1 = nn.ModuleList(conv_gating1)
        # self.conv_gating2 = nn.ModuleList(conv_gating2)
        # self.conv_gating3 = nn.ModuleList(conv_gating3)
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
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = feat[:,channel:,:,:] * gating_weight
                feat_gated_all.append(feat_gated)

            feat_total = torch.cat(feat_gated_all, 1)
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

            # feat1 = torch.cat(feat1, 1)
            # feat2 = torch.cat(feat2, 1)
            # feat3 = torch.cat(feat3, 1)
            # gating_weight1 = sigmoid(conv_gating1(feat1))
            # gating_weight2 = sigmoid(conv_gating2(feat2))
            # gating_weight3 = sigmoid(conv_gating3(feat3))

            # channel = int(feat1.shape[1]/2)
            # feat1_gated = feat1[:,channel:,:,:] * gating_weight1
            # feat2_gated = feat2[:,channel:,:,:] * gating_weight2
            # feat3_gated = feat3[:,channel:,:,:] * gating_weight3

            # feat_total = [feat1_gated, feat2_gated, feat3_gated]
            # feat_temp     = torch.cat(feat_total, 1)
            # feat_temp     = relu(conv(feat_gating))
            # x_temp.append(feat_temp)
        return x_temp

@PIXEL.register_module
class PixelGatingonce_three2(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce_three2, self).__init__()
        self.in_channels     = in_channels
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

@PIXEL.register_module
class PixelGatingonce_three2_plus(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce_three2_plus, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        conv_down            = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i], self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_gating.append(nn.Conv2d(self.in_channels[i], 
                                             2, \
                                             kernel_size=3, padding=1))
                conv_down.append(nn.Conv2d(self.in_channels[i], 
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
                feat = feats[k*(self.gating_seq_len-1)+seq][0] + feats[k*(self.gating_seq_len-1)+seq][1]
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = [feats[k*(self.gating_seq_len-1)+seq][i] * gating_weight[:,i,:,:].unsqueeze(1) \
                            for i in range(2)]
                feat_gated_p = feat_gated[0] + feat_gated[1]
                feat_gated_p = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated_p))
                feat_gated_all.append(feat_gated_p)

            feat_total = feat_gated_all[0] + feat_gated_all[1] + feat_gated_all[2]
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

        return x_temp

@PIXEL.register_module
class PixelGatingonce_three2_sum(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce_three2_sum, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_gating          = []
        conv_down            = []
        sigmoid              = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            sigmoid.append(nn.Sigmoid())
            conv.append(nn.Conv2d(self.in_channels[i], self.in_channels[i],\
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
        t = 0

        for num_in in  range(len(self.in_channels)):
            for seq in range(self.gating_seq_len-1):
                feats.append([x[self.gating_seq_len-1][num_in],x[seq][num_in]])

        for sigmoid, conv, relu in zip(self.sigmoid, self.conv, self.relu):
            
            feat_gated_all = 0
            channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
            for seq in range(self.gating_seq_len-1):
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated = [feat[:,i*channel:(i+1)*channel,:,:] * gating_weight[:,i,:,:].unsqueeze(1) \
                            for i in range(2)]
                feat_gated = torch.cat(feat_gated, 1)
                feat_gated = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated))
                feat_gated_all += feat_gated

            # pdb.set_trace()
            feat_total = feat_gated_all + x[-1][t]
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1
            t+=1

        return x_temp
