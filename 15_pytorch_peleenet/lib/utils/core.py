# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import os
import torch
import shutil
import argparse
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from layers.modules import MultiBoxLoss
from data import COCODetection, VOCDetection, detection_collate, preproc
from configs.CC import Config
from termcolor import cprint
from utils.nms_wrapper import nms
import numpy as np
import math


def set_logger(status):
    if status:
        from logger import Logger
        date = time.strftime("%m_%d_%H_%M") + '_log'
        log_path = './logs/' + date
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
        logger = Logger(log_path)
        return logger
    else:
        pass


def get_min_max_sizes(min_ratio, max_ratio, input_size, mbox_source_num):
    step = int(math.floor(max_ratio - min_ratio) / (mbox_source_num - 2))
    min_sizes = list()
    max_sizes = list()
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(input_size * ratio / 100)
        max_sizes.append(input_size * (ratio + step) / 100)

    if min_ratio == 20:
        min_sizes = [input_size * 10 / 100.] + min_sizes
        max_sizes = [input_size * 20 / 100.] + max_sizes
    else:
        min_sizes = [input_size * 7 / 100.] + min_sizes
        max_sizes = [input_size * 15 / 100.] + max_sizes

    return min_sizes, max_sizes


def anchors(config):
    cfg = dict()
    cfg['feature_maps'] = config.anchor_config.feature_maps
    cfg['min_dim'] = config.input_size
    cfg['steps'] = config.anchor_config.steps
    cfg['min_sizes'], cfg['max_sizes'] = get_min_max_sizes(
        config.anchor_config.min_ratio, config.anchor_config.max_ratio, config.input_size, len(cfg['feature_maps']))
    cfg['aspect_ratios'] = config.anchor_config.aspect_ratios
    cfg['variance'] = [0.1, 0.2]
    cfg['clip'] = True
    return cfg


def init_net(net, cfg, resume_net):
    if cfg.model.init_net and not resume_net:
        net.init_model(cfg.model.pretained_model, cfg.test_cfg.cuda)
    else:
        print('Loading resume network...')
        if(cfg.test_cfg.cuda):
            state_dict = torch.load(resume_net)
        else:
            state_dict = torch.load(resume_net, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict, strict=False)

def init_net2(cfg, resume_net):
    print('Loading resume network...')
    if(cfg.test_cfg.cuda):
        net = torch.load(resume_net)
    else:
        net = torch.load(resume_net, map_location='cpu')

    return net;


def set_optimizer(net, cfg):
    return optim.SGD(net.parameters(),
                     lr=cfg.train_cfg.lr,
                     momentum=cfg.optimizer.momentum,
                     weight_decay=cfg.optimizer.weight_decay)


def set_criterion(cfg):
    return MultiBoxLoss(cfg.model.num_classes,
                        overlap_thresh=cfg.loss.overlap_thresh,
                        prior_for_matching=cfg.loss.prior_for_matching,
                        bkg_label=cfg.loss.bkg_label,
                        neg_mining=cfg.loss.neg_mining,
                        neg_pos=cfg.loss.neg_pos,
                        neg_overlap=cfg.loss.neg_overlap,
                        encode_target=cfg.loss.encode_target)


def adjust_learning_rate(optimizer, step_index, cfg, dataset):
    global lr
    lr = cfg.train_cfg.lr * (0.1**step_index) if cfg.train_cfg.lr * \
        (0.1**step_index) > cfg.train_cfg.end_lr else cfg.train_cfg.end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_dataloader(cfg, train_img_dir=None, train_anno_dir=None, class_file=None):
    _preproc = preproc(cfg.model.input_size, cfg.model.rgb_means, cfg.model.p)
    dataset = VOCDetection(train_img_dir, train_anno_dir, class_file, preproc=_preproc)

    '''
    if setname == 'train_sets':
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                       getattr(cfg.dataset, dataset)[setname], _preproc)
    else:
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                       getattr(cfg.dataset, dataset)[setname], None)
    '''
    return dataset
    


def print_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||Batch_Time:{:.4f}||LR:{:.7f}'.format(*info_list), 'green')


def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def save_checkpoint(net, cfg, final=True, datasetname='COCO', epoch=10):
    weights_save_path = os.path.join(cfg.model.weights_save, datasetname)+'/'
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
    if final:
        torch.save(net.state_dict(), weights_save_path +
                   'Final_Pelee_{}_size{}.pth'.format(datasetname, cfg.model.input_size))
    else:
        torch.save(net.state_dict(), weights_save_path +
                   'Pelee_{}_size{}_epoch{}.pth'.format(datasetname, cfg.model.input_size, epoch))

def save_checkpoint2(net, cfg, final=True, datasetname='COCO', epoch=10):
    weights_save_path = os.path.join(cfg.model.weights_save, datasetname)+'/'
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
    if final:
        torch.save(net, weights_save_path +
                   'Final_Pelee_{}_size{}.pth'.format(datasetname, cfg.model.input_size))
    else:
        torch.save(net, weights_save_path +
                   'Pelee_{}_size{}_epoch{}.pth'.format(datasetname, cfg.model.input_size, epoch))


def write_logger(info_dict, logger, iteration, status):
    if status:
        for tag, value in info_dict.items():
            logger.scalar_summary(tag, value, iteration)
    else:
        pass


def image_forward(img, net, cuda, priors, detector, transform):
    w, h = img.shape[1], img.shape[0]
    scale = torch.Tensor([w, h, w, h])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    return boxes, scores


def nms_process(num_classes, i, scores, boxes, cfg, min_thresh, all_boxes, max_per_image):
    for j in range(1, num_classes):  # ignore the bg(category_id=0)
        inds = np.where(scores[:, j] > min_thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
        # keep only the highest boxes
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1]
                                  for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]
