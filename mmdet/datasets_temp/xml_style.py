import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .custom import CustomDataset


class XMLDataset(CustomDataset):

    def __init__(self, **kwargs):
        super(XMLDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

    def load_annotations(self, ann_file, seq_len):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        seq_delta = [-(seq+1) for seq in range(seq_len)]
        seq_delta.reverse()
        for seq in range(seq_len+1):
            seq_delta.append(seq)
        for img_id in img_ids:            
            if len(osp.basename(img_id)) == 6:
                img_id_seq = [img_id[:-len(osp.basename(img_id))] + \
                              '%06d'%(int(osp.basename(img_id))+seq) \
                              for seq in seq_delta]
            else:
                img_id_seq = [img_id for seq in range(seq_len*2+1)]

            filename_seq = ['JPEGImages/{}.JPEG'.format(img_id_frame) for img_id_frame in \
                            img_id_seq]

            for idx, filename in enumerate(filename_seq):
                if osp.exists(osp.join(self.img_prefix, filename)):
                    continue
                else:
                    if idx < seq_len:
                        if idx == 0:
                            img_id_seq[idx] = img_id_seq[idx+1]
                            filename_seq[idx] = filename_seq[idx+1]
                        else:
                            for idx2 in range(idx+1):
                                img_id_seq[idx2] = img_id_seq[idx+1]
                                filename_seq[idx2] = filename_seq[idx+1]
                    elif idx > seq_len:
                        img_id_seq[idx] = img_id_seq[idx-1]
                        filename_seq[idx] = filename_seq[idx-1]

            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id_seq, filename=filename_seq, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        # import pdb; pdb.set_trace()
        ann_idx = int((len(self.img_infos[idx]['id'])-1)/2)
        img_id = self.img_infos[idx]['id'][ann_idx]
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.cat2label.keys():
                continue
            label = self.cat2label[name]
            if obj.find('difficult') is None:
                difficult = 0
            else:
                difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann