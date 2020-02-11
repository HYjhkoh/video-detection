from .two_stage_lstm_fgating import TwoStageDetectorLSTM_fgating
from ..registry import DETECTORS


@DETECTORS.register_module
class FasterRCNNLSTM_fgating(TwoStageDetectorLSTM_fgating):

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
                 temporal=None,
                 cur=None,
                 in_channels=None):
        super(FasterRCNNLSTM_fgating, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            temporal=temporal,
            cur=cur,
            in_channels=in_channels)
