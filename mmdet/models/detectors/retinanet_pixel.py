from .single_stage_pixel import SingleStageDetectorPixel
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNetPixel(SingleStageDetectorPixel):

    def __init__(self,
                 backbone,
                 neck,
                 temporal,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 gating_seq_len=None):
        super(RetinaNetPixel, self).__init__(backbone, neck, temporal, bbox_head, train_cfg, test_cfg, pretrained, gating_seq_len)
