from .single_stage_corr import SingleStageDetectorCorr
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNetCorr(SingleStageDetectorCorr):

    def __init__(self,
                 backbone,
                 neck,
                 temporal,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 corr_size=None,
                 in_channels=None):
        super(RetinaNetCorr, self).__init__(backbone, neck, temporal, bbox_head, train_cfg,
                                        test_cfg, pretrained, in_channels, corr_size)