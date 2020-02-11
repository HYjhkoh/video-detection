
import pdb
import time
import argparse
import os
import datasets as datasets
import json
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils_aug.augmentations import SSDAugmentation, RetinaAugmentation, NoAugmentation

from model import *
from loss import FocalLoss
from utils import freeze_bn, calculate_mAP
from logger import Logger
from encoder import DataEncoder

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, trainloader, optimizer, device, epoch, criterion, step, logger, batch_size, loss_fn, loss_accum, conf_thres, nms_thres):
    encoder = DataEncoder(loss_fn, conf_thres, nms_thres)
    model.train()
    # freeze_bn(model) # http://qr.ae/TUIS14
    start_time = time.time()
    optimizer.zero_grad()

    batch_loss = 0

    for batch_idx, (seq_data,loc_targets,cls_targets,ori_img_shape, scales, max_shape, ori_shape) in enumerate(trainloader):

        if loss_accum and (batch_idx % loss_accum == 0):
            batch_loc_loss, batch_cls_loss, batch_loss = 0, 0, 0

        inputs, loc_targets, cls_targets = [seq_data[i].to(device).float() for i in range(len(seq_data))],loc_targets.to(device), cls_targets.to(device)
        optimizer.zero_grad()
        
        loc_preds_split, cls_preds_split, loss, loc_loss, cls_loss = model(inputs, loc_targets, cls_targets)
        cls_loss = cls_loss.mean()
        loc_loss = loc_loss.mean()
        loss = loss.mean()
        batch_cls_loss += cls_loss.detach().cpu().numpy() / loss_accum
        batch_loc_loss += loc_loss.detach().cpu().numpy() / loss_accum
        batch_loss += loss.detach().cpu().numpy() / loss_accum
        loss = torch.div(loss, loss_accum)

        loss.backward()
        if loss_accum and (batch_idx % loss_accum == loss_accum - 1):
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 20 == 0:
                end_time = time.time()
                print('[%d,%5d] cls_loss: %.5f loc_loss: %.5f train_loss: %.5f time: %.3f lr: %.6f' % \
                        (epoch, step, batch_cls_loss, batch_loc_loss, batch_loss,  \
                        end_time-start_time, optimizer.param_groups[0]['lr']))
                start_time = time.time()
                info = {'training loss': batch_loss, 'loc_loss': batch_loc_loss, \
                        'cls_loss': batch_cls_loss}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

        if step % 20 == 0:
            pred_boxes, pred_labels, score_all = [], [], []
            if inputs[-1].shape[0] > 10:
                show_num_img = 10
            else:
                show_num_img = inputs[-1].shape[0]
            for img_idx in range(show_num_img):
                max_width, max_height = max_shape
                rows, cols = ori_shape[img_idx]
                loc_preds_batch = [loc_preds_split[i][img_idx,:,:].unsqueeze(0) for i in range(len(loc_preds_split))]
                cls_preds_batch = [cls_preds_split[i][img_idx,:,:].unsqueeze(0) for i in range(len(cls_preds_split))]

                pred_box, pred_label, score = encoder.decode(loc_preds_batch, cls_preds_batch, seq_data[-1].shape, (1,3,max_height,max_width), (3,cols,rows),0)
                pred_boxes.append(pred_box)
                pred_labels.append(pred_label)
                score_all.append(score)
            info = {'images': inputs[-1][:show_num_img].cpu().numpy()}
            for tag, images in info.items():
                images = logger.image_drawbox(images, pred_boxes, pred_labels, score_all)
                logger.image_summary(tag, images, step)
    return step

def test(model, testloader, device, logger, epoch, loss_fn, conf_thres, nms_thres, batch_size, num_classes, label_map, data_name):
    model.eval()
    encoder = DataEncoder(loss_fn, conf_thres, nms_thres)

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        for batch_idx, (seq_data, gt_boxes, gt_labels, difficulties, scales, max_shape, ori_shape) in enumerate(tqdm(testloader, desc='Evaluating')):

            inputs = [seq_data[i].to(device).float() for i in range(len(seq_data))]
            batch_size = inputs[-1].shape[0]
            rows = ori_shape[0][1]
            cols = ori_shape[0][0]
            max_width = max_shape[0]
            max_height = max_shape[1]

            batch_loc_preds = []
            batch_cls_preds = []
            batch_scores = []
            loc_preds_split, cls_preds_split = model(inputs, gt_boxes,\
                                                     gt_labels)
            loc_preds, cls_preds, score = encoder.decode(loc_preds_split, cls_preds_split, seq_data[-1].shape, (1,3,max_width, max_height), (3,cols,rows), 0)
            batch_loc_preds.append(loc_preds.to(device))
            batch_cls_preds.append(cls_preds.to(device).float())
            batch_scores.append(score.to(device))
            # pdb.set_trace()

            boxes = [b.to(device).float() for b in gt_boxes]
            labels = [l.to(device).float() for l in gt_labels]
            difficulties = [d.to(device).float() for d in difficulties]

            det_boxes.extend(batch_loc_preds)
            det_labels.extend(batch_cls_preds)
            det_scores.extend(batch_scores)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        print('==>> Calculate mAP')
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, label_map, device)
        
    if data_name == 'KITTI':
        mAP = APs['car']
    info = {'mAP': mAP}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
    print('mAP: %.5f '%(mAP))

def make_label_map(label_map_txt):
    with open(label_map_txt, 'r') as lmt:
        lines = lmt.readlines()

    label_map = {}
    for l in lines:
        name, idx = l.rstrip('\n').split(' ')[0], int(l.rstrip('\n').split(' ')[1])
        label_map[name] = idx
    return label_map

def save_checkpoint(state, epoch, save_path):
    path = save_path+"/retina_{}.pth".format(epoch)
    torch.save(state,path)
    print("Checkpoint saved to {}".format(path))

def adjust_learning_rate(optimizer, lr, decay_idx, decay_param):
    for lr_idx in range(decay_idx+1):
        lr /= decay_param
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def dataload(trainlist, testlist, data, min_scale, max_scale, transform, batch_size, num_workers):

    print("==>>  Loading the data.....", data)

    trainset = datasets.LoadDataset(trainlist, scale=(min_scale,max_scale),\
                                    shuffle=True, transform=transform, \
                                    train=True, batch_size=batch_size, \
                                    num_workers=num_workers)
    testset = datasets.LoadDataset(testlist, scale=(min_scale,max_scale), \
                                   shuffle=False, transform=transform, \
                                   train=False, batch_size=1, \
                                   num_workers=num_workers)
    # trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, \
            num_workers=num_workers, collate_fn=trainset.collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=testset.collate_fn)

    return trainloader, testloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights','-w', type=str, default='False')
    parser.add_argument('--debug','-d', type=str, default='False')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num-workers', '-n', type=int, default=os.cpu_count())
    parser.add_argument('--config', '-c', type=str, default='')
    args = parser.parse_args()

    if args.config is None:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example")

    config = json.load(open(args.config))

    os.system('python3 model.py -c %s' % args.config)

    data        = config['data']['name']
    conf_thres  = config['data']['conf_thres']
    nms_thres   = config['data']['nms_thres']
    num_workers = args.num_workers
    trainlist   = config['data']['train_dir']
    testlist    = config['data']['test_dir']

    batch_size  = config['trainer']['batch_size']
    loss_accum  = config['trainer']['loss_accum']
    loss        = config['trainer']['loss']
    opt         = config['trainer']['optimizer']
    lr          = config['trainer']['learning_rate']

    base        = config['network']['base']
    network    = config['network']['name']

    use_cuda    = torch.cuda.is_available()
    step        = 0

    if data == 'ILSVRC':
        min_scale = 500
        max_scale = 832
        gpus = [0,1]
        label_map = make_label_map('./label_map/%s.txt'%data)
        if loss == "sigmoid":
            num_classes = 200
        elif loss == "softmax":
            num_classes = 201
        total_iter = 90000*12

    elif data == 'KITTI':
        min_scale = 384
        max_scale = 1248
        gpus = [0, 1]
        label_map = make_label_map('./label_map/%s.txt'%data)
        if loss == "sigmoid":
            num_classes = 3
        elif loss == "softmax":
            num_classes = 4
        total_iter = 90000/193*200

    lr_decay_param = 0

    save_path = './weights_%s_%s/' % (lr, network)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.debug == 'True':
        num_workers = 0
        gpus = [0] 

    ##### Data Loading #####
    if config['data']['aug'] == 'ssd':
        transform = SSDAugmentation()
    elif config['data']['aug'] == 'retina':
        transform = RetinaAugmentation()
    elif config['data']['aug'] == 'none':
        transform = NoAugmentation()

    # dataset
    trainloader, testloader = dataload(trainlist, testlist, data, \
                                       min_scale,max_scale, transform, \
                                       batch_size, num_workers)

    ##### Training Setup #####

    total_epoch = int(total_iter/len(trainloader.dataset)*batch_size)
    print('==>>  Total_epoch size is %d'%(total_epoch))

    decay_epochs = [int(total_epoch*2/3),int(total_epoch*8/9)]
    decay_param = 10
    decay_idx = 0

    if args.debug == 'True':
        logger = Logger('./logs_debug')
    else:
        logger = Logger('./logs_%s_%s' %(lr, network))
    
    seed = int(time.time())
    torch.manual_seed(seed)
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    # setting network
    if base == 'resnet':
        if network == 'FPN18':
            model = ResNet(num_classes, network, BasicBlock)
        else:
            model = ResNet(num_classes, network)
    elif base == 'resnext':
        model = ResNeXt(num_classes, network)

    # setting optimizer
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        print('==>>wrong opt name')

    # setting loss
    criterion = FocalLoss(num_classes, loss) # nn.CrossEntropyLoss()

    checkpoint = torch.load('./init_weight/net_%s_%s_%s_%s.pt'%(data,loss,network,base))
    model.load_state_dict(checkpoint)

    keys = model.state_dict().keys()
    for name, child in model.named_children():
        if not name.startswith('lstm') and not name.startswith('regressionModel') and not name.startswith('classificationModel'):
            pdb.set_trace()
            for param in child.parameters():
                param.requires_grad = False
                
    if use_cuda:
        if len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        else:
            model = model.to(device)
        model.cuda()

    if args.weights:
        if os.path.isfile(args.weights):
            print("==>>  Loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            args.start_epoch = checkpoint['epoch']+1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==>>  Loaded checkpoint '{}' (epoch {}))".format(args.weights,checkpoint['epoch']))
        else:
            print("==>>  No checkpoint found")

    # train
    for epoch in range(args.start_epoch, total_epoch):
        
        step = train(model, trainloader, optimizer, device, epoch, \
                     criterion, step, logger, batch_size, loss, \
                     loss_accum, conf_thres, nms_thres)
        if epoch % 2 == 0:
            save_checkpoint({'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()},epoch,save_path)
        if epoch % 5 == 0:
            test(model, testloader, device, logger, epoch, loss, \
                 conf_thres, nms_thres, batch_size, num_classes, \
                 label_map, data)

        if lr_decay_param == 0:
            if epoch == decay_epochs[decay_idx]:
                adjust_learning_rate(optimizer, lr, decay_idx, decay_param)
                decay_idx += 1
                if len(decay_epochs) == decay_idx:
                    lr_decay_param = 1

if __name__ == '__main__':
    main()
