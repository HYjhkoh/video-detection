from .two_stage_temp_rtemp_many_relu import TwoStageDetectorTempRTempManyRelu
from ..registry import DETECTORS


@DETECTORS.register_module
class FasterRCNNTempRTempManyRelu(TwoStageDetectorTempRTempManyRelu):

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
                 temporal2=None):
        super(FasterRCNNTempRTempManyRelu, self).__init__(
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
            temporal2=temporal2)
