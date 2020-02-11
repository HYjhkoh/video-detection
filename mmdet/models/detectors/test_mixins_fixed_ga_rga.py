from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)


class RPNTestMixin(object):

    def simple_test_rpn(self, x_temp, x_all, img_meta, rpn_test_cfg):

        seq_len = len(x_all)
        cur_idx = int((seq_len-1)/2)
       
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

        proposal_inputs = rpn_out_cur + (img_meta, rpn_test_cfg, [0], 'True')
        proposal_list_ori, inds = self.rpn_head.get_bboxes(*proposal_inputs)

        proposal_lists = []
        for seq_idx in range(seq_len):
            if seq_idx != cur_idx:
                proposal_inputs = rpn_outs[seq_idx] + (img_meta, rpn_test_cfg, inds, 'False')
                proposal_list, _ = self.rpn_head.get_bboxes(*proposal_inputs)
                proposal_lists.append(proposal_list)
            else:
                proposal_lists.append(proposal_list_ori)        

        return proposal_lists

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x_temp,
                           x_all,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        seq_len = len(x_all)
        cur_idx = int((seq_len-1)/2)

        rois = []
        for seq_idx in range(seq_len):
            roi = bbox2roi(proposals[seq_idx])
            rois.append(roi)

        # rois = bbox2roi(proposals)

        roi_feats = []
        for seq_idx in range(seq_len):
            if seq_idx != cur_idx:
                roi_feat = self.bbox_roi_extractor(
                    x_all[seq_idx][:self.bbox_roi_extractor.num_inputs], rois[seq_idx])
            else:
                roi_feat = self.bbox_roi_extractor(
                    x_temp[:self.bbox_roi_extractor.num_inputs], rois[seq_idx])                
            roi_feats.append(roi_feat)

        bbox_feats = self.temporal2(roi_feats)[0]

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois[cur_idx],
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
