import torch.nn as nn
from torch.autograd import Variable
import torch

from .base_temp import BaseDetectorTemp
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result

import pdb

@DETECTORS.register_module
class SingleStageDetectorPixelLSTM(BaseDetectorTemp):

    def __init__(self,
                 backbone,
                 neck=None,
                 pixel = None,
                 motion = None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 cur=None,
                 gating_seq_len=None):
        super(SingleStageDetectorPixelLSTM, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.pixel = builder.build_pixel(pixel)
        self.motion = builder.build_motion(motion)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gating_seq_len = gating_seq_len
        self.init_weights(pretrained=pretrained)
        self.cur = cur

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorPixelLSTM, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights(pretrained=pretrained)
        self.pixel.init_weights(pretrained)
        self.motion.init_weights()
        self.bbox_head.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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
        for i in range(len(x_all[0])):
            batch_size, channel, height, width = x_all[0][i].shape
            init_state = (Variable(torch.zeros(batch_size, channel, height, width)).cuda(), 
                          Variable(torch.zeros(batch_size, channel, height, width)).cuda())
            multi_init_state.append(init_state)
            
        if self.cur == True:
            for i in range(seq_len):
                multi_init_state = self.motion(multi_init_state, x_all[i])
        else:
            for i in range(seq_len-1):
                multi_init_state = self.motion(multi_init_state, x_all[i])  
        
        x_motion = [multi_init_state[i][0] for i in range(len(x_all[0]))]

        x_pixel = self.pixel([x_all[seq_len-self.gating_seq_len+i] for i in range(self.gating_seq_len)])

        # import pdb; pdb.set_trace()
        outs = self.bbox_head(x_pixel, x_motion, x_all[-1])
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
        _,_,seq_len,_,_ = img.shape
        x_all = []
        for i in range(seq_len):
            x = self.extract_feat(img[:,:,i,:,:])
            x_all.append(x)
        
        multi_init_state = []
        for i in range(len(x_all[0])):
            batch_size, channel, height, width = x_all[0][i].shape
            init_state = (Variable(torch.zeros(batch_size, channel, height, width)).cuda(), 
                          Variable(torch.zeros(batch_size, channel, height, width)).cuda())
            multi_init_state.append(init_state)
            
        for i in range(seq_len-1):
            multi_init_state = self.motion(multi_init_state, x_all[i], x_all[i+1])
        
        x_motion = [multi_init_state[i][0] for i in range(len(x_all[0]))]

        x_pixel = self.pixel([x_all[seq_len-self.gating_seq_len+i] for i in range(self.gating_seq_len)])

        # import pdb; pdb.set_trace()
        outs = self.bbox_head(x_pixel, x_motion, x_all[-1])

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
