from .single_stage_motion import SingleStageDetectorMotion
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNetMotion(SingleStageDetectorMotion):

    def __init__(self,
                 backbone,
                 neck,
                 temporal,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetMotion, self).__init__(backbone, neck, temporal, bbox_head, train_cfg,
                                        test_cfg, pretrained)
