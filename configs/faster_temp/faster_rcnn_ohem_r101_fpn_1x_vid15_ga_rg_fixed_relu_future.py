# model settings
model = dict(
    type='FasterRCNNTempRTempMany',
    pretrained='pretrained/faster_101_batch8_ohem_56.pth',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPNRelu',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    temporal=dict(
        type='PixelGatingAlignFuture',
        in_channels=(256, 256, 256, 256, 256),
        gating_seq_len=5,
        corr_size=5),
    temporal2=dict(
        type='PixelGatingonce_three2_future_roi',
        in_channels=(256, 256, 256, 256, 256),
        gating_seq_len=5),
    rpn_head=dict(
        type='RPNHeadFixed',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=31,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='OHEMSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'VIDDataset'
data_root = '../dataset/ILSVRC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        # ann_file=[
        #         data_root + 'DET/ImageSets/Main/train_30cls.txt',
        #         data_root + 'VID/ImageSets/Main/train_15frames.txt'
        #         ],
        ann_file=[
                data_root + 'VID/ImageSets/Main/train_1img.txt'
                ],
        seq_len=2,
        # img_prefix=[data_root + 'DET/train/',data_root + 'VID/train/'],
        img_prefix=[data_root + 'VID/train/'],
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VID/ImageSets/Main/val_15frames.txt',
        seq_len=2,
        img_prefix=data_root + 'VID/val/',
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VID/ImageSets/Main/val.txt',
        seq_len=2,
        img_prefix=data_root + 'VID/val/',
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 10
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_temp/faster_rcnn_r101_fpn_1x_vid15_ohem_8_ga_rg_fixed_relu_future'
load_from = None
resume_from = './work_dirs/faster_temp/faster_rcnn_r101_fpn_1x_vid15_ohem_8_ga_rg_fixed_relu_future/epoch_2.pth'
workflow = [('train', 1)]