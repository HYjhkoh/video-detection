import time

import numpy as np
import spconv
import torch
from torch import nn
from torch.nn import functional as F
import pdb
from . import resnet_seg
# from second.pytorch.models.resnet_seg import *
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.utils import torch_timer

REGISTERED_FUSED_CLASSES = {}

def register_fused(cls, name=None):
    global REGISTERED_FUSED_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_FUSED_CLASSES, f"exist class: {REGISTERED_FUSED_CLASSES}"
    REGISTERED_FUSED_CLASSES[name] = cls
    return cls

def get_fused_class(name):
    global REGISTERED_FUSED_CLASSES
    assert name in REGISTERED_FUSED_CLASSES, f"available class: {REGISTERED_FUSED_CLASSES}"
    return REGISTERED_FUSED_CLASSES[name]


@register_fused
class FusedFeatureExtractor(nn.Module):
    def __init__(self): #, crit, deep_sup_scale=None):
        super(FusedFeatureExtractor, self).__init__()
        self.encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
        # self.decoder = net_dec
        # self.crit = crit
        # self.deep_sup_scale = deep_sup_scale
        self.crop_feature_channel_reduce = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.fusion_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

    def _feature_crop(self, feature, idx_c):
        '''cropping from projection coordinates 
        '''
        grid_num = 2
        num_coord = idx_c.shape[1]
        a1 = torch.tensor([[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5, 0.5]])

        batch_size = feature.shape[0]
        f_w, f_h = feature.shape[2], feature.shape[3]
        crop_feature = torch.zeros(batch_size, feature.shape[1], num_coord).cuda()
        crop_feature_ori = torch.zeros(batch_size, feature.shape[1], num_coord).cuda()

        crop_feature1 = torch.zeros(batch_size, feature.shape[1], num_coord*grid_num**2).cuda()

        for i in range(idx_c.shape[0]):
            idx = idx_c[i].clone()
            # Filtering idx
            mask_ori = torch.mul(idx >= 0, idx <= 1).sum(dim=1) != 2
            idx[mask_ori] = 0
            idx_upsamp = idx*torch.tensor([f_w,f_h]).view(1,2).cuda().to(torch.float32)
            rep_coord = idx_upsamp.repeat_interleave(grid_num**2,dim=0)
            rep_a1 = a1.repeat(num_coord,1).cuda()
            c_coord = torch.floor(rep_coord+rep_a1) ## minus debug!
            cen_coord = c_coord+0.5
            rep_mask = mask_ori.repeat_interleave(grid_num**2, dim=0)
            w_coord = ((cen_coord-rep_coord)**2).sum(1).sqrt()
            w_norm = w_coord[0::4] + w_coord[1::4] + w_coord[2::4] + w_coord[3::4]
            w_norm = w_norm.repeat_interleave(grid_num**2,dim=0)
            w_coord = w_coord/w_norm
            w_coord[rep_mask] = 0
            c_coord[rep_mask,:] = 0
            mask = torch.mul(c_coord[:,0] >= 0, c_coord[:,0] < f_w) + torch.mul(c_coord[:,1] >= 0, c_coord[:,1] < f_h) != 2
            w_coord[mask] = 0
            c_coord[mask, :] = 0

            crop_feature1[i,:,:] = feature[i,:,c_coord[:,0].to(torch.int64), c_coord[:,1].to(torch.int64)] * w_coord.view(1,num_coord*grid_num**2)

            crop_feature[i,:,:] = crop_feature1[i,:,:][:,0::4] + crop_feature1[i,:,:][:,1::4] + crop_feature1[i,:,:][:,2::4] + crop_feature1[i,:,:][:,3::4]
            crop_feature_ori[i,:,:] = feature[i, :, (idx[:, 0] * f_w).to(torch.int64), (idx[:, 1] * f_h).to(torch.int64)]
            crop_feature_ori[i, :, mask_ori] = 0
            
        crop_features_cc = crop_feature.reshape(batch_size, -1, 200, 176)
        crop_features_cc_ori = crop_feature_ori.reshape(batch_size, -1, 200, 176)
        # pdb.set_trace()

        return crop_features_cc, crop_features_cc_ori

    def forward(self, bev_feature, f_view, idxs_norm):

        ###################################################################
        # RGB Backbone Network (Resnet18)rr
        # f1 = self.maxpool(F.relu(self.bn1(self.conv1(f_view))))
        # f2 = self.layer1(f1)
        # f3 = self.layer2(f2)
        # f4 = self.layer3(f3)
        # f5 = self.layer4(f4)
        # f_view_features = self.fpn([f3, f4, f5])
        f_view_features = self.encoder(f_view, return_feature_maps=True)
        # pred = self.decoder(, segSize=segSize)
        fuse_features = F.relu(f_view_features[0]).permute(0,1,3,2)
        fuse_features = fuse_features.permute(0,1,3,2)

        # Feature Transform
        crop_feature2, crop_feature_ori = self._feature_crop(fuse_features, idxs_norm)


        ###################################################################
        import cv2
        import os
        fol_name = 'proposed2'
        img_id = 0
        layer_list = []
        
        layer_list.append(f_view_features[0].cpu().detach().numpy())
        layer_list.append(crop_feature2.cpu().detach().numpy())
        layer_list.append(bev_feature.cpu().detach().numpy())

        # layer_list.append(crop_feature.cpu().detach().numpy())
        # layer_list.append(crop_feature2.cpu().detach().numpy())
        # layer_list.append(bev_feature.cpu().detach().numpy())
        # layer_list.append(bev_feature2.cpu().detach().numpy())
        layer_name = ['fuse_rgb','crop', 'bev'] #,'crop', 'bev', 'bev_n']


        for img_id in range(2):
            input_view = f_view.cpu().detach().numpy()
            input_view = np.array(input_view[img_id,:,:,:])
            input_view = np.transpose(input_view,(1,2,0))

            input_view = input_view + np.abs(np.min(input_view))
            input_view = input_view/np.max(input_view)*255

            for idx in range(len(layer_list)) :
                ori_run = np.array(layer_list[idx])
                if 'fuse' not in layer_name[idx]:
                    ori_run = np.transpose(ori_run,(0,3,2,1))
                    ori_run = np.flip(ori_run, axis=1)
                else :
                    ori_run = np.transpose(ori_run,(0,2,3,1))

                layer_tot_ori = []
                for idx3 in range(int(ori_run.shape[3]/32)) :
                    layer_con_ori = []
                    for idx2 in range(32):
                        ori_img = cv2.resize(ori_run[img_id,:,:,idx2+32*idx3],dsize=(0, 0),fx=1, fy=1,interpolation=cv2.INTER_NEAREST)
                        ori_img = ori_img/np.max(ori_img)*255
                        ori_img = np.array(ori_img, np.uint8)
                        ori_img = cv2.applyColorMap(ori_img, cv2.COLORMAP_JET)
                        const_ori= cv2.copyMakeBorder(ori_img,1,1,1,1,cv2.BORDER_CONSTANT,value=[255,0,0])
                        layer_con_ori.append(const_ori)
                    layer_con_ori = np.concatenate(layer_con_ori,axis=1)
                    layer_tot_ori.append(layer_con_ori)
                layer_tot_ori = np.concatenate(layer_tot_ori,axis=0)
                if not os.path.exists('./'+fol_name+'/img'+str(img_id)):
                    os.makedirs('./'+fol_name+'/img'+str(img_id))
                cv2.imwrite('./'+fol_name+'/img'+str(img_id)+'/'+layer_name[idx].replace('/','_')+'.png', layer_tot_ori)
                cv2.imwrite('./'+fol_name+'/img'+str(img_id)+'/input_image.png', input_view)
        pdb.set_trace()

        crop_feature = self.crop_feature_channel_reduce(crop_feature2)

        # fused_feature = torch.cat((bev_feature, crop_feature), dim=1)
        fused_feature = torch.cat((bev_feature, bev_feature), dim=1)
        fused_feature = F.relu(self.fusion_conv(fused_feature))
        
        fused_feature = torch.cat((bev_feature, fused_feature), dim=1)

        return fused_feature

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet_seg.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = resnet_seg(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet_seg.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet_seg.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = resnet_seg(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet_seg.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet_seg.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = resnet_seg(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet_seg.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet_seg.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = resnet_seg(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet_seg.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = resnet_seg(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
