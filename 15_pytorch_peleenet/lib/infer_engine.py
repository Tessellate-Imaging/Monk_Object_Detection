import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from peleenet import build_net
from data import BaseTransform, VOC_CLASSES
from utils.core import *
from utils.pycocotools.coco import COCO


class Infer():
    '''
    Class for main inference

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};

    def Data_Params(self, class_list):
        '''
        User Function - Set Dataset params

        Args:
            class_list (list): List of classes in the same order as training
        
        Returns:
            None
        '''
        self.system_dict["class_list"] = class_list;

    def _to_color(self, indx, base):
        """ return (b, r, g) tuple"""
        base2 = base * base
        b = 2 - indx / base2
        r = 2 - (indx % base2) / base
        g = 2 - (indx % base2) % base
        return b * 127, r * 127, g * 127

    def draw_detection(self, im, bboxes, scores, cls_inds, fps, thr=0.2):
        imgcv = np.copy(im)
        h, w, _ = imgcv.shape
        for i, box in enumerate(bboxes):
            if scores[i] < thr:
                continue
            cls_indx = int(cls_inds[i])
            box = [int(_) for _ in box]
            thick = int((h + w) / 300)
            cv2.rectangle(imgcv,
                          (box[0], box[1]), (box[2], box[3]),
                          self.system_dict["colors"][cls_indx], thick)
            mess = '%s: %.3f' % (self.system_dict["labels"][cls_indx], scores[i])
            cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                        0, 1e-3 * h, self.system_dict["colors"][cls_indx], thick // 3)
            if fps >= 0:
                cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15),
                            0, 2e-3 * h, (255, 255, 255), thick // 2)

        return imgcv


    def Model_Params(self, model_dir="output", use_gpu=True):
        '''
        User Function - Set Model Params

        Args:
            model_dir (str): Select the right model name as per training
            model_path (str): Relative path to params file
            use_gpu (bool): If True use GPU else run on CPU
        Returns:
            None
            
        '''

        f = open(model_dir +"/config_final.py", 'r');
        lines = f.read();
        f.close();

        if(not use_gpu):
            lines = lines.replace("cuda=True",
                                    "cuda=False");

        f = open(model_dir +"/config_test.py", 'w');
        f.write(lines);
        f.close();


        print("Loading model for inference");
        self.system_dict["cfg"] = Config.fromfile(model_dir +"/config_test.py")
        anchor_config = anchors(self.system_dict["cfg"].model)
        self.system_dict["priorbox"] = PriorBox(anchor_config)
        self.system_dict["net"] = build_net('test', self.system_dict["cfg"].model.input_size, self.system_dict["cfg"].model)
        init_net(self.system_dict["net"], self.system_dict["cfg"], model_dir + "/VOC/Final_Pelee_VOC_size304.pth")
        print_info('===> Finished constructing and loading model', ['yellow', 'bold'])
        self.system_dict["net"].eval()

        with torch.no_grad():
            self.system_dict["priors"] = self.system_dict["priorbox"].forward()
            if self.system_dict["cfg"].test_cfg.cuda:
                self.system_dict["net"] = self.system_dict["net"].cuda()
                self.system_dict["priors"] = self.system_dict["priors"].cuda()
                cudnn.benchmark = True
            else:
                self.system_dict["net"] = self.system_dict["net"].cpu()
        self.system_dict["_preprocess"] = BaseTransform(self.system_dict["cfg"].model.input_size, 
                                        self.system_dict["cfg"].model.rgb_means, (2, 0, 1))
        self.system_dict["num_classes"] = self.system_dict["cfg"].model.num_classes
        self.system_dict["detector"] = Detect(self.system_dict["num_classes"],
                                                self.system_dict["cfg"].loss.bkg_label, anchor_config)
                
        print("Done....");


        print("Loading other params");
        base = int(np.ceil(pow(self.system_dict["num_classes"], 1. / 3)))
        self.system_dict["colors"] = [self._to_color(x, base)
                  for x in range(self.system_dict["num_classes"])]
        cats = ['__background__'];
        f = open(self.system_dict["class_list"]);
        lines = f.readlines();
        f.close();
        for i in range(len(lines)):
            if(lines != ""):
                cats.append(lines[i][:len(lines[i])-1])
        self.system_dict["labels"] = cats;
        print("Done....");

    def Predict(self, im_path, thresh=0.5, visualize=False, output_img_path="output.jpg"):
        loop_start = time.time()
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
        w, h = image.shape[1], image.shape[0]
        img = self.system_dict["_preprocess"](image).unsqueeze(0)
        if self.system_dict["cfg"].test_cfg.cuda:
            img = img.cuda()
        scale = torch.Tensor([w, h, w, h])
        out = self.system_dict["net"](img)
        boxes, scores = self.system_dict["detector"].forward(out, self.system_dict["priors"])
        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, self.system_dict["num_classes"]):
            inds = np.where(scores[:, j] > self.system_dict["cfg"].test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            soft_nms = self.system_dict["cfg"].test_cfg.soft_nms
            # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = nms(c_dets, self.system_dict["cfg"].test_cfg.iou, force_cpu=soft_nms)
            keep = keep[:self.system_dict["cfg"].test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist() + [j] for _ in c_dets])

        loop_time = time.time() - loop_start
        print("Inference time 2 - {} sec".format(loop_time));

        allboxes = np.array(allboxes)
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]
        im2show = self.draw_detection(image, boxes, scores, cls_inds, -1, thresh)

        if im2show.shape[0] > 1100:
            im2show = cv2.resize(im2show,
                                 (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
        if visualize:
            cv2.imshow('test', im2show)
            cv2.waitKey(2000)

        cv2.imwrite(output_img_path, im2show)

    
