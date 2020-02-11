from .xml_style import XMLDataset


class DETDataset(XMLDataset):

    file = open('./mmdet/datasets/map_det.txt', 'r')
    lines = file.readlines()
    label_maps = {}
    for line in lines:
      label = line.rstrip().split()
      label_maps[label[0]] = label[-1]

    CLASSES = [i for i in label_maps.keys()]

    def __init__(self, **kwargs):
        super(DETDataset, self).__init__(**kwargs)

class VIDDataset(XMLDataset):

    file = open('./mmdet/datasets/map_vid.txt', 'r')
    lines = file.readlines()
    label_maps = {}
    for line in lines:
      label = line.rstrip().split()
      label_maps[label[0]] = label[-1]

    CLASSES = [i for i in label_maps.keys()]

    def __init__(self, **kwargs):
        super(VIDDataset, self).__init__(**kwargs)