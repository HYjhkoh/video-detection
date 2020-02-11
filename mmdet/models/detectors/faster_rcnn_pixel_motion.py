from .two_stage_pixel_motion import TwoStageDetectorPixelMotion
from ..registry import DETECTORS


@DETECTORS.register_module
class FasterRCNNPixelMotion(TwoStageDetectorPixelMotion):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 pixel=None,
                 motion=None,
                 corr_size=None,
                 in_channels=None):
        super(FasterRCNNPixelMotion, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            pixel=pixel,
            motion=motion,
            corr_size=corr_size,
            in_channels=in_channels)
