import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins_fixed_ga_rga import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, build_sampler_fixed

import pdb
import cv2
import numpy as np

@DETECTORS.register_module
class TwoStageDetectorTempRTempMany2(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 temporal=None,
                 temporal2=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetectorTempRTempMany2, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
        self.temporal = builder.build_temporal(temporal)
        self.temporal2 = builder.build_temporal(temporal2)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorTempRTempMany2, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights(pretrained=pretrained)
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        self.temporal.init_weights(pretrained=pretrained)
        self.temporal2.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def feature_vis(self, img, features, name):

        inputs = []
        for i in range(img.shape[2]):
            input_view = img[:,:,i,:,:].cpu().detach().numpy()
            input_view = np.array(input_view[0,:,:,:])
            input_view = np.transpose(input_view,(1,2,0))
            input_view[:,:,0] = input_view[:,:,0]*58.395 + 123.675
            input_view[:,:,1] = input_view[:,:,1]*57.12 + 116.28
            input_view[:,:,2] = input_view[:,:,2]*57.375 + 103.53

            # for idx in range(len(input_view.shape[2])):
            inputs.append(input_view)

        layer_con_ori = []
        # for idx in range(32):
        channel_param = 32
        if len(features) == 5:
            seq_len = 5
            for idx in range(seq_len):
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
                cv2.imwrite('./vis_feature/roi_pooling/feature_%s_%d.png'%(name, idx),layer_tot_ori)
            #     layer_con_ori.append(const_ori)
            # layer_con_ori = np.concatenate(layer_con_ori,axis=1)
            # layer_tot_ori.append(layer_con_ori)
            # layer_tot_ori = np.concatenate(layer_tot_ori,axis=0)
            #     if not os.path.exists('./'+fol_name+'/img'+str(img_id)):
            #         os.makedirs('./'+fol_name+'/img'+str(img_id))
            cv2.imwrite('./vis_feature/roi_pooling/input_image1.png', inputs[0])
            cv2.imwrite('./vis_feature/roi_pooling/input_image2.png', inputs[1])
            cv2.imwrite('./vis_feature/roi_pooling/input_image3.png', inputs[2])
            cv2.imwrite('./vis_feature/roi_pooling/input_image4.png', inputs[3])
            cv2.imwrite('./vis_feature/roi_pooling/input_image5.png', inputs[4])
            pdb.set_trace()
        else:
            seq_len = 1
            for idx in range(seq_len):
                feature = features.cpu().detach().numpy()
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
                cv2.imwrite('./vis_feature/roi_pooling/feature_%s_%d.png'%(name, idx),layer_tot_ori)
            #     layer_con_ori.append(const_ori)
            # layer_con_ori = np.concatenate(layer_con_ori,axis=1)
            # layer_tot_ori.append(layer_con_ori)
            # layer_tot_ori = np.concatenate(layer_tot_ori,axis=0)
            #     if not os.path.exists('./'+fol_name+'/img'+str(img_id)):
            #         os.makedirs('./'+fol_name+'/img'+str(img_id))
            cv2.imwrite('./vis_feature/roi_pooling/input_image1.png', inputs[0])
            cv2.imwrite('./vis_feature/roi_pooling/input_image2.png', inputs[1])
            cv2.imwrite('./vis_feature/roi_pooling/input_image3.png', inputs[2])
            cv2.imwrite('./vis_feature/roi_pooling/input_image4.png', inputs[3])
            cv2.imwrite('./vis_feature/roi_pooling/input_image5.png', inputs[4])
            pdb.set_trace()

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        _,_,seq_len,_,_ = img.shape
        cur_idx = int((seq_len-1)/2)
        x_all = []
        for i in range(seq_len):
            x = self.extract_feat(img[:,:,i,:,:])
            x_all.append(x)

        x_temp = self.temporal(x_all)

        # if True:
        #     self.feature_vis(img, x_temp) ## x_all[3], feats, x_temp

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = []

            rpn_out_cur = self.rpn_head(x_temp)
            for seq_idx in range(seq_len):
                if seq_idx != cur_idx:
                    rpn_out = self.rpn_head(x_all[seq_idx])
                    for i in range(len(rpn_out_cur[0])):
                        rpn_out[0][i] = rpn_out_cur[0][i]
                else:
                    rpn_out = rpn_out_cur
                rpn_outs.append(rpn_out)

            # for seq_idx in range(seq_len):
            #     if seq_idx != cur_idx:
            #         rpn_out = self.rpn_head(x_all[seq_idx])
            #     else:
            #         rpn_out = self.rpn_head(x_temp)
            #     rpn_outs.append(rpn_out)

            rpn_loss_inputs = rpn_outs[cur_idx] + (gt_bboxes, img_meta,
                                              self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs[cur_idx] + (img_meta, proposal_cfg, [0], 'True')
            proposal_list_ori, inds = self.rpn_head.get_bboxes(*proposal_inputs)

            proposal_lists = []
            for seq_idx in range(seq_len):
                if seq_idx != cur_idx:
                    proposal_inputs = rpn_outs[seq_idx] + (img_meta, proposal_cfg, inds, 'False')
                    proposal_list, _ = self.rpn_head.get_bboxes(*proposal_inputs)
                    proposal_lists.append(proposal_list)
                else:
                    proposal_lists.append(proposal_list_ori)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            bbox_sampler_fixed = build_sampler_fixed(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

        #     sampling_results_ori = []
        #     for i in range(num_imgs):
        #         assign_result_ori = bbox_assigner.assign(
        #             proposal_lists[cur_idx][i], gt_bboxes[i], gt_bboxes_ignore[i],
        #             gt_labels[i])
        #         sampling_result_ori = bbox_sampler.sample(
        #             assign_result_ori,
        #             proposal_lists[cur_idx][i],
        #             gt_bboxes[i],
        #             gt_labels[i],
        #             feats=[lvl_feat[i][None] for lvl_feat in x_temp])
        #         sampling_results_ori.append(sampling_result_ori)

        #     # import pdb; pdb.set_trace()
        #     seq_sampling_results = []
        #     for seq_idx in range(seq_len):
        #         sampling_results = []
        #         if seq_idx != cur_idx:
        #             for i in range(num_imgs):
        #                 sampling_result = bbox_sampler_fixed.sample(
        #                     assign_result_ori,
        #                     proposal_lists[seq_idx][i],
        #                     gt_bboxes[i],
        #                     gt_labels[i],
        #                     sampling_result_ori[i].pos_inds,
        #                     sampling_result_ori[i].neg_inds,
        #                     feats=[lvl_feat[i][None] for lvl_feat in x_temp])
        #                 sampling_results.append(sampling_result)
        #         else:
        #             seq_sampling_results.append(sampling_results_ori)
        #         seq_sampling_results.append(sampling_results)
            

            sampling_results_ori = []
            assign_results_ori = []
            for i in range(num_imgs):
                assign_result_ori = bbox_assigner.assign(
                    proposal_lists[cur_idx][i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_ori = bbox_sampler.sample(
                    assign_result_ori,
                    proposal_lists[cur_idx][i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x_temp])
                assign_results_ori.append(assign_result_ori)
                sampling_results_ori.append(sampling_result_ori)

            seq_sampling_results = []
            for seq_idx in range(seq_len):
                sampling_results = []
                for i in range(num_imgs):
                    if seq_idx != cur_idx:
                        sampling_result = bbox_sampler_fixed.sample(
                            assign_results_ori[i],
                            proposal_lists[seq_idx][i],
                            gt_bboxes[i],
                            gt_labels[i],
                            sampling_results_ori[i].pos_inds,
                            sampling_results_ori[i].neg_inds,
                            feats=[lvl_feat[i][None] for lvl_feat in x_temp])
                    else:
                        sampling_result = sampling_results_ori[i]
                    sampling_results.append(sampling_result)
                seq_sampling_results.append(sampling_results)

        # bbox head forward and loss
        if self.with_bbox:
            rois = []
            for seq_idx in range(seq_len):
                roi = bbox2roi([res.bboxes for res in seq_sampling_results[seq_idx]])
                rois.append(roi)
            
            # TODO: a more flexible way to decide which feature maps to use
            
            roi_feats = []
            for seq_idx in range(seq_len):
                if seq_idx != cur_idx:
                    roi_feat = self.bbox_roi_extractor(
                        x_all[seq_idx][:self.bbox_roi_extractor.num_inputs], rois[seq_idx])
                    roi_feats.append(roi_feat)
                else:
                    roi_feat = self.bbox_roi_extractor(
                        x_temp[:self.bbox_roi_extractor.num_inputs], rois[seq_idx])
                    roi_feats.append(roi_feat)

            pdb.set_trace()
            if True:
                self.feature_vis(img, roi_feats[4], 'roifeat+2') ## x_all[3], feats, x_temp

            bbox_feats = self.temporal2(roi_feats)[0]

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                seq_sampling_results[cur_idx], gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        _,_,seq_len,_,_ = img.shape
        x_all = []
        for i in range(seq_len):
            x = self.extract_feat(img[:,:,i,:,:])
            x_all.append(x)

        x_temp = self.temporal(x_all)

        proposal_lists = self.simple_test_rpn(
            x_temp, x_all, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(x_temp,
            x_all, img_meta, proposal_lists, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
