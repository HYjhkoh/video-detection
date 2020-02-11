import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import AnchorGenerator, anchor_target, multi_apply
from .anchor_head import AnchorHead
from ..losses import smooth_l1_loss

from mmcv.runner import load_checkpoint
from ..registry import HEADS


# TODO: add loss evaluator for SSD
@HEADS.register_module
class SSDHeadMove(AnchorHead):

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        reg_convs  = []
        move_convs = []
        cls_convs  = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            move_convs.append(
                nn.Conv2d(
                    9,
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
        self.reg_convs  = nn.ModuleList(reg_convs)
        self.move_convs = nn.ModuleList(move_convs)
        self.cls_convs  = nn.ModuleList(cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]  # 4 or 6 ratio
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

    def init_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

        if pretrained=='pretrained/ssd300_vid.pth':
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, feats, feats_pre, feats_move, return_loss):
        if return_loss:
            cls_scores = []
            bbox_preds = []
            move_preds = []
            for feat, feat_move, reg_conv, cls_conv, move_conv in zip(feats, feats_move, self.reg_convs, self.cls_convs, self.move_convs):
                cls_scores.append(cls_conv(feat))
                bbox_preds.append(reg_conv(feat))
                move_preds.append(move_conv(feat_move))

            return cls_scores, bbox_preds, move_preds
        else:
            cls_scores     = []
            bbox_preds     = []
            cls_scores_pre = []
            bbox_preds_pre = []
            move_preds     = []
            for feat, feat_pre, feat_move, reg_conv, cls_conv, move_conv in zip(feats, feats_pre, feats_move, self.reg_convs, self.cls_convs, self.move_convs):
                cls_scores.append(cls_conv(feat))
                bbox_preds.append(reg_conv(feat))
                cls_scores_pre.append(cls_conv(feat_pre))
                bbox_preds_pre.append(reg_conv(feat_pre))
                move_preds.append(move_conv(feat_move))

            return cls_scores, bbox_preds, cls_scores_pre, bbox_preds_pre, move_preds

    def loss_single(self, cls_score, bbox_pred, move_pred, labels_pre, labels, label_weights_pre, label_weights, bbox_targets_pre, bbox_targets,bbox_weights_pre,  bbox_weights, num_total_samples_pre, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        move_targets = bbox_targets - bbox_targets_pre
        loss_move = smooth_l1_loss(
            move_pred,
            move_targets,
            bbox_weights_pre,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox, loss_move

    def loss(self,
             cls_scores,
             bbox_preds,
             move_preds,
             gt_bboxes_pre,
             gt_bboxes,
             gt_labels_pre,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list_pre, valid_flag_list_pre = self.get_anchors(
            featmap_sizes, img_metas)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        cls_reg_targets_pre = anchor_target(
            anchor_list_pre,
            valid_flag_list_pre,
            gt_bboxes_pre,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels_pre,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        (labels_list_pre, label_weights_list_pre, bbox_targets_list_pre, bbox_weights_list_pre,
         num_total_pos_pre, num_total_neg_pre) = cls_reg_targets_pre

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels_pre = torch.cat(labels_list_pre, -1).view(num_images, -1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights_pre = torch.cat(label_weights_list_pre,
                                      -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)

        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_move_preds = torch.cat([
            m.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for m in move_preds
        ], -2)
        all_bbox_targets_pre = torch.cat(bbox_targets_list_pre,
                                     -2).view(num_images, -1, 4)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights_pre = torch.cat(bbox_weights_list_pre,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        losses_cls, losses_bbox, losses_move = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_move_preds,
            all_labels_pre,
            all_labels,
            all_label_weights_pre,
            all_label_weights,
            all_bbox_targets_pre,
            all_bbox_targets,
            all_bbox_weights_pre,
            all_bbox_weights,
            num_total_samples_pre=num_total_pos_pre,
            num_total_samples=num_total_pos,
            cfg=cfg)
        
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_move=losses_move)
