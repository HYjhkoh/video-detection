import torch
import torch.nn as nn
from torch.autograd import Variable

from .base_temp import BaseDetectorTemp
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetectorTemp(BaseDetectorTemp):

    def __init__(self,
                 backbone,
                 neck=None,
                 temporal = None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetectorTemp, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.temporal = builder.build_temporal(temporal)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorTemp, self).init_weights(pretrained)
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

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # self.debug_train(img[1], gt_bboxes[1], gt_labels[1])
        x = []
        seq_len = img.shape[2]
        batch_size = len(gt_bboxes)
        for i in range(seq_len):
            x_single = self.extract_feat(img[:,:,i,:,:])
            x.append(x_single)

        multi_init_state = []
        for i in range(len(x[0])):
            batch_size, channel, height, width = x[0][i].shape
            init_state = (Variable(torch.zeros(batch_size, channel, height, width)).cuda(),\
                          Variable(torch.zeros(batch_size, channel, height, width)).cuda())
            multi_init_state.append(init_state)

        multi_previous_state = multi_init_state
        for seq in range(seq_len):
            multi_previous_state = self.temporal(x[seq], multi_previous_state)

        import pdb; pdb.set_trace()
        x_temp = [h_state for h_state, _ in multi_previous_state]
        outs = self.bbox_head(x_temp)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def debug_train(self, img, gt_bboxes, gt_labels):
        import cv2
        for i in range(img.shape[1]):
            img_single = img[:,i,:,:].cpu().permute(1,2,0).numpy()
            img_box = img_single.copy()
            for j in range(gt_bboxes.shape[0]):
                start = (int(gt_bboxes[j][0]), int(gt_bboxes[j][1]))
                end = (int(gt_bboxes[j][2]), int(gt_bboxes[j][3]))
                color=(0,0,255)
                img_box = cv2.rectangle(img_box, start, end, color,1)
            cv2.imwrite('%d.png'%i, img_box)
        import pdb; pdb.set_trace()

    
    def debug(self, img, gt_bboxes, gt_labels):
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
        x = []
        seq_len = img.shape[2]
        batch_size = 1
        for i in range(seq_len):
            x_single = self.extract_feat(img[:,:,i,:,:])
            x.append(x_single)

        multi_init_state = []
        for i in range(len(x[0])):
            batch_size, channel, height, width = x[0][i].shape
            init_state = (Variable(torch.zeros(batch_size, channel, height, width)).cuda(),\
                          Variable(torch.zeros(batch_size, channel, height, width)).cuda())
            multi_init_state.append(init_state)

        multi_previous_state = multi_init_state
        for seq in range(seq_len):
            multi_previous_state = self.temporal(x[seq], multi_previous_state)

        x_temp = [h_state for h_state, _ in multi_previous_state]
        outs = self.bbox_head(x_temp)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        # import pdb; pdb.set_trace()
        # self.debug_test(img_pre[0], img[0], bbox_results[0][26], img_meta)
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
