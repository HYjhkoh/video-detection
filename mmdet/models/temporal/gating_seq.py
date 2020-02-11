import torch
import torch.nn as nn

from .. import builder
from ..registry import TEMPORAL
from mmcv.cnn import xavier_init

import logging
from mmcv.runner import load_checkpoint


@TEMPORAL.register_module
class GatingSeq(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 seq_len=4):
        super(GatingSeq, self).__init__()
        self.in_channels = in_channels
        self.seq_len     = seq_len
        conv_gating      = []
        sigmoid          = []
        conv             = []
        relu             = []

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
            x_all.append([x[seq][num_in] for seq in range(self.seq_len)])

        for feat, conv_gating, sigmoid, conv, relu in zip(x_all, self.conv_gating, self.sigmoid, self.conv, self.relu):
            
            feat_cat      = torch.cat(feat, 1)
            gating_weight = sigmoid(conv_gating(feat_cat))
            feat_gating   = [feat[i] * gating_weight[:,i,:,:].unsqueeze(1) \
                             for i in range(len(feat))]

            feat_temp     = torch.cat(feat_gating, 1)
            feat_temp     = relu(conv(feat_temp))
            x_temp.append(feat_temp)

        return x_temp


@TEMPORAL.register_module
class GatingSeq_single(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 seq_len=4):
        super(GatingSeq_single, self).__init__()
        self.in_channels = in_channels
        self.seq_len     = seq_len
        conv_gating      = []
        sigmoid          = []
        conv             = []
        relu             = []

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
            x_all.append([x[seq][num_in] for seq in range(self.seq_len)])

        for feat, conv_gating, sigmoid, conv, relu in zip(x_all, self.conv_gating, self.sigmoid, self.conv, self.relu):
            
            feat_temp = feat[-1]
            # feat_cat      = torch.cat(feat, 1)
            # gating_weight = sigmoid(conv_gating(feat_cat))
            # feat_gating   = [feat[i] * gating_weight[:,i,:,:].unsqueeze(1) \
            #                  for i in range(len(feat))]

            # feat_temp     = torch.cat(feat_gating, 1)
            # feat_temp     = relu(conv(feat_temp))
            x_temp.append(feat_temp)

        return x_temp

@TEMPORAL.register_module
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


@TEMPORAL.register_module
class Pixelconcat(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(Pixelconcat, self).__init__()
        self.in_channels     = in_channels
        self.gating_seq_len  = gating_seq_len
        conv_down            = []
        conv                 = []
        relu                 = []

        for i in range(len(self.in_channels)):
            conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
                                  kernel_size=3, padding=1))
            relu.append(nn.ReLU(inplace=True))
            for seq in range(gating_seq_len-1):
                conv_down.append(nn.Conv2d(self.in_channels[i]*2, 
                                             self.in_channels[i], \
                                             kernel_size=3, padding=1))                    

        self.conv_down = nn.ModuleList(conv_down)
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

        for conv, relu in zip(self.conv, self.relu):
            
            feat_gated_all = []
            channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
            for seq in range(self.gating_seq_len-1):
                feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
                feat = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat))
                feat_gated_all.append(feat)

            feat_total = torch.cat(feat_gated_all, 1)
            feat_temp = relu(conv(feat_total))
            x_temp.append(feat_temp)
            k+=1

        return x_temp

@TEMPORAL.register_module
class PixelGatingonce_three2_roi(nn.Module):

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 gating_seq_len=4):
        super(PixelGatingonce_three2_roi, self).__init__()
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
        for seq in range(self.gating_seq_len-1):
            feats.append([x[self.gating_seq_len-1],x[seq]])
        

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

# @TEMPORAL.register_module
# class PixelGatingAlign(nn.Module):

#     def __init__(self,
#                  in_channels=(512, 1024, 512, 256, 256, 256),
#                  corr_size=5,
#                  gating_seq_len=4):
#         super(PixelGatingAlign, self).__init__()
#         self.in_channels     = in_channels
#         self.corr_size       = corr_size
#         self.gating_seq_len  = gating_seq_len
#         conv_gating          = []
#         conv_down            = []
#         sigmoid              = []
#         conv                 = []
#         relu                 = []

#         for i in range(len(self.in_channels)):
#             sigmoid.append(nn.Sigmoid())
#             conv.append(nn.Conv2d(self.in_channels[i]*(gating_seq_len-1), self.in_channels[i],\
#                                   kernel_size=3, padding=1))
#             relu.append(nn.ReLU(inplace=True))
#             for seq in range(gating_seq_len-1):
#                 conv_gating.append(nn.Conv2d(self.in_channels[i]*2, 
#                                              2, \
#                                              kernel_size=3, padding=1))
#                 conv_down.append(nn.Conv2d(self.in_channels[i]*2, 
#                                              self.in_channels[i], \
#                                              kernel_size=3, padding=1))                 
                                             

#         self.conv_gating = nn.ModuleList(conv_gating)
#         self.conv_down = nn.ModuleList(conv_down)
#         self.sigmoid = nn.ModuleList(sigmoid)
#         self.conv = nn.ModuleList(conv)
#         self.relu = nn.ModuleList(relu)

#     def init_weights(self, pretrained):
#         if pretrained == None:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     xavier_init(m, distribution='uniform', bias=0)

#         else:
#             if 'gating' in pretrained:
#                 # import pdb; pdb.set_trace()
#                 for m in self.modules():
#                     if isinstance(m, nn.Conv2d):
#                         xavier_init(m, distribution='uniform', bias=0)
#                 logger = logging.getLogger()
#                 load_checkpoint(self, pretrained, strict=False, logger=logger)

#             else:
#                 for m in self.modules():
#                     if isinstance(m, nn.Conv2d):
#                         xavier_init(m, distribution='uniform', bias=0)

#     def forward(self, x):

#         feats = []
#         x_temp = []
#         k = 0

#         for num_in in  range(len(self.in_channels)):
#             for seq in range(self.gating_seq_len-1):
#                 feats.append([x[self.gating_seq_len-1][num_in],x[seq][num_in]])

#         for sigmoid, conv, relu in zip(self.sigmoid, self.conv, self.relu):
            
#             feat_gated_all = []

#             channel = feats[k*(self.gating_seq_len-1)][0].shape[1]
#             for seq in range(self.gating_seq_len-1):
#                 [feat2, feat1] = feats[k*(self.gating_seq_len-1)+seq]
#                 corr_feat = spatial_correlation_sample(feat2,
#                                                        feat1,
#                                                        kernel_size=1, 
#                                                        patch_size=self.corr_size,
#                                                        stride=1,
#                                                        padding=0,
#                                                        dilation_patch=1)

#                 b, ph, pw, h, w = corr_feat.shape
#                 corr_feat = F.softmax(corr_feat.view(b,ph*pw,h,w)*20/(corr_feat.max()))
#                 corr_feat = corr_feat.view(b,1,ph,pw,h,w)
#                 pad_size = int((self.corr_size-1)/2)
#                 pad_feat1 = F.pad(feat1, (pad_size,pad_size,pad_size,pad_size))
                
#                 for i in range(self.corr_size):
#                     for j in range(self.corr_size):
#                         mul_feat = corr_feat[:,:,i,j,:,:] * pad_feat1[:,:,i:h+i,j:j+w]
#                         if i == 0 and j == 0:
#                             new_feat = mul_feat
#                         else:
#                             new_feat += mul_feat
#                 feats[k*(self.gating_seq_len-1)+seq][-1] = new_feat
                
#                 feat = torch.cat(feats[k*(self.gating_seq_len-1)+seq],1)
#                 gating_weight = sigmoid(self.conv_gating[k*(self.gating_seq_len-1)+seq](feat))
#                 feat_gated = [feat[:,i*channel:(i+1)*channel,:,:] * gating_weight[:,i,:,:].unsqueeze(1) \
#                             for i in range(2)]
#                 feat_gated = torch.cat(feat_gated, 1)
#                 feat_gated = relu(self.conv_down[k*(self.gating_seq_len-1)+seq](feat_gated))
#                 feat_gated_all.append(feat_gated)

#             feat_total = torch.cat(feat_gated_all, 1)
#             feat_temp = relu(conv(feat_total))
#             x_temp.append(feat_temp)
#             k+=1

#         return x_temp