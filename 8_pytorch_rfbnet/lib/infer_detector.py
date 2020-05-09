import sys
import os
import pickle
import argparse
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer


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
        self.system_dict["params"] = {};

        self.system_dict["params"]["dataset"] = "COCO";
        self.system_dict["params"]["version"] = "RFB_vgg";
        self.system_dict["params"]["size"] = 512;
        self.system_dict["params"]["cuda"] = True;
        self.system_dict["params"]["cpu"] = False;
        self.system_dict["params"]["retest"] = False;

        self.system_dict["params"]["trained_model"] = "";
        self.system_dict["params"]["save_folder"] = "eval";

        self.system_dict["local"]["img_file"] = "";
        self.system_dict["local"]["thresh"] = 0.3;
        self.system_dict["local"]["classes"] = ["__bg__"];


    def Model(self, model_name="vgg", weights="weights/Final_RFB_vgg_COCO.pth", use_gpu=True):
        '''
        User function: Selet trained model params

        Args:
            model_name (str): Select the right model
            weights (str): Relative path to the trained model 
            use_gpu (bool): If True, model is loaded on GPU else cpu

        Returns:
            None
        '''
        if(model_name == "vgg"):
            self.system_dict["params"]["version"] = "RFB_vgg";
        elif(model_name == "e_vgg"):
            self.system_dict["params"]["version"] = "RFB_E_vgg";
        elif(model_name == "mobilenet"):
            self.system_dict["params"]["version"] = "RFB_mobile";

        self.system_dict["params"]["trained_model"] = weights;        

        if(use_gpu):
            self.system_dict["params"]["cuda"] = True;
            self.system_dict["params"]["cpu"] = False;
        else:
            self.system_dict["params"]["cuda"] = False;
            self.system_dict["params"]["cpu"] = True;


    def Image_Params(self, class_file, input_size=512):
        '''
        User function: Set trained model params

        Args:
            class_file (str): Path to file containing class names
            input_size (int): Input image shape

        Returns:
            None
        '''
        self.system_dict["params"]["size"] = input_size;
        f = open(class_file, 'r');
        lines = f.readlines();
        f.close();
        for i in range(len(lines)):
            self.system_dict["local"]["classes"].append(lines[i][:len(lines[i])-1])


    def Setup(self):
        '''
        User function: Setup all parameters

        Args:
            None

        Returns:
            None
        '''
        if(self.system_dict["params"]["size"] == 300):
            self.system_dict["local"]["cfg"] = COCO_300;
        else:
            self.system_dict["local"]["cfg"] = COCO_512;


        if self.system_dict["params"]["version"] == 'RFB_vgg':
            from models.RFB_Net_vgg import build_net
        elif self.system_dict["params"]["version"] == 'RFB_E_vgg':
            from models.RFB_Net_E_vgg import build_net
        elif self.system_dict["params"]["version"] == 'RFB_mobile':
            from models.RFB_Net_mobile import build_net
            self.system_dict["local"]["cfg"] = COCO_mobile_300
        else:
            print('Unkown version!')


        self.system_dict["local"]["priorbox"] = PriorBox(self.system_dict["local"]["cfg"])
        with torch.no_grad():
            self.system_dict["local"]["priors"] = self.system_dict["local"]["priorbox"].forward()
            if self.system_dict["params"]["cuda"]:
                self.system_dict["local"]["priors"] = self.system_dict["local"]["priors"].cuda()


        img_dim = (300,512)[self.system_dict["params"]["size"] == 512]
        num_classes = len(self.system_dict["local"]["classes"])
        self.system_dict["local"]["net"] = build_net('test', img_dim, num_classes)
        state_dict = torch.load(self.system_dict["params"]["trained_model"])


        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.system_dict["local"]["net"].load_state_dict(new_state_dict)
        self.system_dict["local"]["net"].eval()
        print('Finished loading model!')

        if self.system_dict["params"]["cuda"]:
            self.system_dict["local"]["net"] = self.system_dict["local"]["net"].cuda()
            cudnn.benchmark = True
        else:
            self.system_dict["local"]["net"] = self.system_dict["local"]["net"].cpu()

        
        self.system_dict["local"]["detector"] = Detect(num_classes, 0, self.system_dict["local"]["cfg"])



    
    def Predict(self, img_file, thresh=0.7, font_size=1, line_size=5):
        '''
        User function: Run inference on image and visualize it

        Args:
            img_file (str): Relative path to the image file
            thresh (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
            font_size (int): Font size for text of label names on predicted images
            line_size (int): Drawn bounding boxes line widths

        Returns:
            list: List of bounding box locations of predicted objects along with classes. 
        '''
        top_k = 200
        img_dim = (300,512)[self.system_dict["params"]["size"] == 512]
        num_classes = len(self.system_dict["local"]["classes"])

        rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[self.system_dict["params"]["version"] == 'RFB_mobile']

        cuda = self.system_dict["params"]["cuda"];
        transform = BaseTransform(self.system_dict["local"]["net"].size, rgb_means, (2, 0, 1))
        max_per_image = top_k
        
        num_images = 1
        all_boxes = [];

        _t = {'im_detect': Timer(), 'misc': Timer()}
        det_file = os.path.join('detections.pkl')

        
        img_id = img_file;
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img_id, cv2.IMREAD_COLOR)

        scale = torch.Tensor([img.shape[1], img.shape[0],
                     img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = self.system_dict["local"]["net"](x)      # forward pass
        boxes, scores = self.system_dict["local"]["detector"].forward(out, self.system_dict["local"]["priors"])
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()

        classes = self.system_dict["local"]["classes"];

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=self.system_dict["params"]["cpu"])
            c_dets = c_dets[keep, :]
            c_bboxes=c_dets[:, :4]
            print(len(c_bboxes), j)
            for bbox in c_bboxes:
                cv2.rectangle(img2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), line_size)
                cv2.putText(img2, classes[j], (int(bbox[0]), int(bbox[1])) , 
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), line_size-2)
                all_boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), classes[j]])
            
        cv2.imwrite("output.png", img2)

        return all_boxes;

