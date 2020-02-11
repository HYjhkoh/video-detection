import torch.nn as nn

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
                      img_pre,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x_pre = self.extract_feat(img_pre)
        x = self.extract_feat(img)
        x_temp = self.temporal(x_pre, x)
        outs = self.bbox_head(x_temp)
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

    
    def simple_test(self, img_pre, img, img_meta, rescale=False):
        x_pre = self.extract_feat(img_pre)
        x = self.extract_feat(img)
        x_temp = self.temporal(x_pre, x)
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
