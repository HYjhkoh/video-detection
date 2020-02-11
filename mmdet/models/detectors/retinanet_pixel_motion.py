from .single_stage_pixel_motion import SingleStageDetectorPixelMotion
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNetPixelMotion(SingleStageDetectorPixelMotion):

    def __init__(self,
                 backbone,
                 neck,
                 pixel,
                 motion,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 corr_size=None,
                 gating_seq_len=None):
        super(RetinaNetPixelMotion, self).__init__(backbone, neck, pixel, motion, bbox_head, train_cfg,
                                        test_cfg, pretrained, corr_size, gating_seq_len)
