import torch.nn as nn
from torch.autograd import Variable
import torch

from .base_temp import BaseDetectorTemp
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result
from mmcv.cnn import xavier_init

import numpy as np
import cv2
import pdb

@DETECTORS.register_module
class SingleStageDetectorCorr_flow(BaseDetectorTemp):

    def __init__(self,
                 backbone,
                 neck=None,
                 temporal = None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 in_channels=None,
                 corr_size=None):
        super(SingleStageDetectorCorr_flow, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.temporal = builder.build_temporal(temporal)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.in_channels = in_channels
        self.corr_size = corr_size

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorCorr_flow, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.temporal.init_weights()
        self.bbox_head.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def feature_vis(self, img, features):

        inputs = []
        for i in range(img.shape[2]):
            input_view = img[:,:,i,:,:].cpu().detach().numpy()
            input_view = np.array(input_view[0,:,:,:])
            input_view = np.transpose(input_view,(1,2,0))
            inputs.append(input_view)

        # layer_con_ori = []
        # for idx in range(32):
        channel_param = 3
        for idx in range(len(features)):
            feature = features[idx].cpu().detach().numpy()
            feature = np.transpose(feature,(0,2,3,1))
            layer_tot_ori = []
            for idx3 in range(int(feature.shape[3]/channel_param)):
                layer_con_ori = []
                for idx2 in range(channel_param):
                    ori_img = cv2.resize(feature[0,:,:,idx2+channel_param*idx3],dsize=(0, 0),fx=1, fy=1,interpolation=cv2.INTER_NEAREST)
                    ori_img = ori_img/np.max(ori_img)*255
                    ori_img = np.array(ori_img, np.uint8)
                    ori_img = cv2.applyColorMap(ori_img, cv2.COLORMAP_JET)
                    const_ori= cv2.copyMakeBorder(ori_img,1,1,1,1,cv2.BORDER_CONSTANT,value=[255,0,0])
                    layer_con_ori.append(const_ori)
                layer_con_ori = np.concatenate(layer_con_ori,axis=1)
                layer_tot_ori.append(layer_con_ori)
            layer_tot_ori = np.concatenate(layer_tot_ori,axis=0)
            cv2.imwrite('./vis_feature/corr/feature_out_%d.png'%(idx),layer_tot_ori)
            # layer_con_ori.append(const_ori)
        # layer_con_ori = np.concatenate(layer_con_ori,axis=1)
        # layer_tot_ori.append(layer_con_ori)
        # layer_tot_ori = np.concatenate(layer_tot_ori,axis=0)
            # if not os.path.exists('./'+fol_name+'/img'+str(img_id)):
            #     os.makedirs('./'+fol_name+'/img'+str(img_id))
        cv2.imwrite('./vis_feature/corr/input_image1.png', inputs[0])
        cv2.imwrite('./vis_feature/corr/input_image2.png', inputs[1])
        cv2.imwrite('./vis_feature/corr/input_image3.png', inputs[2])
        cv2.imwrite('./vis_feature/corr/input_image4.png', inputs[3])
        pdb.set_trace()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        _,_,seq_len,_,_ = img.shape
        x_all = []
        for i in range(seq_len):
            x = self.extract_feat(img[:,:,i,:,:])
            x_all.append(x)

        multi_init_state = []
        for i in range(len(self.in_channels)):
            batch_size, _, height, width = x_all[0][i].shape
            init_state = (Variable(torch.zeros(batch_size, int(self.corr_size*self.corr_size+self.in_channels[i]/8), height, width)).cuda(),\
                          Variable(torch.zeros(batch_size, int(self.corr_size*self.corr_size+self.in_channels[i]/8), height, width)).cuda())
            multi_init_state.append(init_state)

        for i in range(seq_len-1):
            multi_init_state, corr_feats = self.temporal(x_all[i], x_all[i+1], multi_init_state)

        multi_init_state = [multi_init_state[i][0] for i in range(len(self.in_channels))]

        # if True:
        #     self.feature_vis(img, multi_init_state)

        outs = self.bbox_head(multi_init_state)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def debug(self, img_pre, img, gt_bboxes, gt_labels):
        import cv2
        img_pre = img_pre.cpu().permute(1,2,0).numpy()
        img = img.cpu().permute(1,2,0).numpy()
        start = (int(gt_bboxes[0][0]), int(gt_bboxes[0][1]))
        end = (int(gt_bboxes[0][2]), int(gt_bboxes[0][3]))
        color=(0,0,255)
        img_pre_box = cv2.rectangle(img_pre, start, end, color,1)
        img_box = cv2.rectangle(img, start, end, color,1)
        cv2.imwrite('a.png', img_pre_box)
        cv2.imwrite('b.png', img_box)
        import pdb; pdb.set_trace()

    
    def simple_test(self, img, img_meta, rescale=False):
        # import pdb; pdb.set_trace()
        _,_,seq_len,_,_ = img.shape
        x_all = []
        for i in range(seq_len):
            x = self.extract_feat(img[:,:,i,:,:])
            x_all.append(x)

        multi_init_state = []
        for i in range(len(self.in_channels)):
            batch_size, _, height, width = x_all[0][i].shape
            init_state = (Variable(torch.zeros(batch_size, int(self.corr_size*self.corr_size+self.in_channels[i]/8), height, width)).cuda(),\
                          Variable(torch.zeros(batch_size, int(self.corr_size*self.corr_size+self.in_channels[i]/8), height, width)).cuda())
            multi_init_state.append(init_state)

        for i in range(seq_len-1):
            multi_init_state, corr_feats = self.temporal(x_all[i], x_all[i+1], multi_init_state)

        multi_init_state = [multi_init_state[i][0] for i in range(len(self.in_channels))]

        # if True:
        #     self.feature_vis(img, multi_init_state)

        outs = self.bbox_head(multi_init_state)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def debug_test(self, img_pre, img, boxes, img_meta):
        import cv2
        (h,w,c) = img_meta[0]['ori_shape']
        img_pre = img_pre.cpu().permute(1,2,0).numpy()
        img_pre = cv2.resize(img_pre, (w,h))
        img = img.cpu().permute(1,2,0).numpy()
        img = cv2.resize(img, (w,h))
        start = (int(boxes[0][0]), int(boxes[0][1]))
        end = (int(boxes[0][2]), int(boxes[0][3]))
        color=(0,0,255)
        img_box = cv2.rectangle(img, start, end, color,1)
        img_pre_box = cv2.rectangle(img_pre, start, end, color,1)
        cv2.imwrite('a.png', img_pre_box)
        cv2.imwrite('b.png', img_box)
        import pdb; pdb.set_trace()

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

