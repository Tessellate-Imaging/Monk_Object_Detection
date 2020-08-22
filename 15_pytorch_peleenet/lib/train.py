#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import shutil
import argparse
from peleenet import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *

parser = argparse.ArgumentParser(description='Pelee Training')
parser.add_argument('-c', '--config', default='configs/Pelee_VOC.py')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool,
                    default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       Pelee Training Program                       |\n'
           '----------------------------------------------------------------------', ['yellow', 'bold'])

logger = set_logger(args.tensorboard)
global cfg
cfg = Config.fromfile(args.config)
net = build_net('train', cfg.model.input_size, cfg.model)
init_net(net, cfg, args.resume_net)  # init the network with pretrained
if args.ngpu > 1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benckmark = True

optimizer = set_optimizer(net, cfg)
criterion = set_criterion(cfg)
priorbox = PriorBox(anchors(cfg.model))

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()


if __name__ == '__main__':
    net.train()
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...', ['yellow', 'bold'])
    dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = cfg.train_cfg.step_lr[-1] + 1

    stepvalues = cfg.train_cfg.step_lr

    print_info('===> Training STDN on ' + args.dataset, ['yellow', 'bold'])

    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0
    step_index = 0
    for step in stepvalues:
        if start_iter > step:
            step_index += 1

    for iteration in xrange(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset,
                                                  cfg.train_cfg.per_batch_size * args.ngpu,
                                                  shuffle=True,
                                                  num_workers=cfg.train_cfg.num_workers,
                                                  collate_fn=detection_collate))
            if epoch % cfg.model.save_epochs == 0:
                save_checkpoint(net, cfg, final=False,
                                datasetname=args.dataset, epoch=epoch)
            epoch += 1
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(
            optimizer, step_index, cfg, args.dataset)
        images, targets = next(batch_iterator)
        if cfg.train_cfg.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        write_logger({'loc_loss': loss_l.item(),
                      'conf_loss': loss_c.item(),
                      'loss': loss.item()}, logger, iteration, status=args.tensorboard)
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        print_train_log(iteration, cfg.train_cfg.print_epochs,
                        [time.ctime(), epoch, iteration % epoch_size, epoch_size, iteration, loss_l.item(), loss_c.item(), load_t1 - load_t0, lr])

    save_checkpoint(net, cfg, final=True,
                    datasetname=args.dataset, epoch=-1)
