import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

def debug(img, boxes, labels,name):
    import cv2
    num = len(labels)
    for i in range(num):
        start = (int(boxes[i][0]), int(boxes[i][1]))
        end = (int(boxes[i][2]), int(boxes[i][3]))
        color=(0,0,255)
        img_box = cv2.rectangle(img, start, end, color,3)
    cv2.imwrite('%s.png'%name, img_box)
    import pdb; pdb.set_trace()

class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        seq_len = len(boxes)
        if random.randint(2):    
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = np.asarray([mmcv.bgr2hsv(img[i, ...]) for i in range(seq_len)])

        # random saturation
        if random.randint(2):
            img[:,:,:,1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[:,:,:,0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[:,:,:,0][img[:,:,:,0] > 360] -= 360
            img[:,:,:,0][img[:,:,:,0] < 0] += 360

        # convert color from HSV to BGR
        img = np.asarray([mmcv.hsv2bgr(img[i, ...]) for i in range(seq_len)])

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[:,:,:,random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        t, h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((t, int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[:, top:top + h, left:left + w, :] = img
        img = expand_img
        boxes = [boxes[i] + np.tile((left, top), 2) for i in range(t)]
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        t, h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))

                overlaps_min = []
                for seq in range(t):
                    overlaps = bbox_overlaps(
                        patch.reshape(-1, 4), boxes[seq].reshape(-1, 4)).reshape(-1)
                    overlaps_min.append(overlaps.min())
                
                if np.array(overlaps_min).min() < min_iou:
                    continue      

                # center of boxes should inside the crop img
                mask_all = []
                mask_any = []
                
                for seq in range(t):       
                    center = (boxes[seq][:, :2] + boxes[seq][:, 2:]) / 2
                    mask = (center[:, 0] > patch[0]) * (
                        center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                            center[:, 1] < patch[3])
                    mask_all.append(mask)
                    mask_any.append(mask.any())

                if not np.array(mask_any).all():
                    continue

                # adjust boxes
                img = img[:,patch[1]:patch[3], patch[0]:patch[2],:]
                for seq in range(t):
                    boxes[seq] = boxes[seq][mask_all[seq]]
                    labels[seq] = labels[seq][mask_all[seq]]
                    boxes[seq][:, 2:] = boxes[seq][:, 2:].clip(max=patch[2:])
                    boxes[seq][:, :2] = boxes[seq][:, :2].clip(min=patch[:2])
                    boxes[seq] -= np.tile(patch[:2], 2)

                return img, boxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, labels):

        img = np.asarray(img, dtype=np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        # debug(img[0,:,:,:], boxes[0], labels[0], 'pre')
        # debug(img[1,:,:,:], boxes[1], labels[1], 'cur')

        return img, boxes, labels