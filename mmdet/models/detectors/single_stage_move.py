import torch.nn as nn

from .base_temp import BaseDetectorTemp
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result
from spatial_correlation_sampler import spatial_correlation_sample


@DETECTORS.register_module
class SingleStageDetectorMove(BaseDetectorTemp):

    def __init__(self,
                 backbone,
                 neck=None,
                 temporal = None,
                 bbox_head_move=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 seq_len=None):
        super(SingleStageDetectorMove, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.temporal = builder.build_temporal(temporal)
        self.bbox_head_move = builder.build_head(bbox_head_move)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.seq_len = seq_len
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorMove, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.temporal.init_weights()
        self.bbox_head_move.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_pre,
                      img_metas,
                      return_loss,
                      gt_bboxes_pre,
                      gt_bboxes,
                      gt_labels_pre,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        x_pre = self.extract_feat(img_pre)
        x_move = self.temporal(x_pre, x)
        outs = self.bbox_head_move(x, x_pre, x_move, return_loss)
        loss_bbox_inputs = outs + (gt_bboxes_pre, gt_bboxes, gt_labels_pre, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head_move.loss(
            *loss_bbox_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def debug(self, img_pre, img, gt_bboxes, gt_labels):
        import cv2
        box_num = gt_labels.shape[0]
        img_pre_box = img_pre.cpu().permute(1,2,0).numpy()
        img_box = img.cpu().permute(1,2,0).numpy()
        for i in range(box_num):
            start = (int(gt_bboxes[i][0]), int(gt_bboxes[i][1]))
            end = (int(gt_bboxes[i][2]), int(gt_bboxes[i][3]))
            color=(0,0,255)
            img_pre_box = cv2.rectangle(img_pre_box, start, end, color,1)
            img_box = cv2.rectangle(img_box, start, end, color,1)
        cv2.imwrite('a.png', img_pre_box)
        cv2.imwrite('b.png', img_box)
        # import pdb; pdb.set_trace()

    
    def simple_test(self, img_pre, img, img_meta, return_loss, rescale=False):

        x_pre = self.extract_feat(img_pre)
        x = self.extract_feat(img)
        x_move = self.temporal(x_pre, x)
        outs = self.bbox_head_move(x, x_pre, x_move, return_loss)
        import pdb; pdb.set_trace()
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head_move.get_bboxes(*bbox_inputs)
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
