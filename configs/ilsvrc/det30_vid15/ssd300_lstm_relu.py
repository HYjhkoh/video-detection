# model settings
input_size = 300
model = dict(
    type='SingleStageDetectorTemp',
    pretrained='pretrained/ssd300_det.pth',
    backbone=dict(
        type='SSDVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=None,
    temporal=dict(
        type='ConvLSTM_relu',
        in_channels=(512, 1024, 512, 256, 256, 256)),
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=31,
        anchor_strides=(8, 16, 32, 64, 100, 300),
        basesize_ratio_range=(0.2, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'VIDDataset'
data_root = '../dataset/ILSVRC/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'DET/ImageSets/Main/train_30cls.txt',
                data_root + 'VID/ImageSets/Main/train_15frames.txt'
            ],
            seq_len=4,
            img_prefix=[data_root + 'DET/train/',data_root + 'VID/train/'],
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VID/ImageSets/Main/val_sample.txt',
        seq_len=4,
        img_prefix=data_root + 'VID/val/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VID/ImageSets/Main/val.txt',
        seq_len=4,
        img_prefix=data_root + 'VID/val/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
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
total_epochs = 120
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/det30_vid15/ssd300_lstm_relu'
load_from = None
resume_from = ''
workflow = [('train', 1)]
