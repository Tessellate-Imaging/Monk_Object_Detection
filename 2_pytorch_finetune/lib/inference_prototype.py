import os
import sys
import numpy as np
import pandas as pd
import cv2

from engine import train_one_epoch, evaluate
import utils
import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.notebook import tqdm
import transforms as T
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.autograd import Variable

class Infer():
    '''
    Class for main inference

    Args:
        model_name (str): Select the right model name as per training
        params_file (str): Relative path to params file
        class_list (list): List of classes in the same order as training
        use_gpu (bool): If True use GPU else run on CPU
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, model_name, params_file, class_list, use_gpu=True, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["classes"] = class_list;
        self.system_dict["local"] = {};
        self.system_dict["model_set_1"] = ["faster-rcnn_mobilenet-v2"]

        self.load_model(model_name, params_file, use_gpu=use_gpu);


    def load_model(self, model_name, params_file, use_gpu=True):
        '''
        Internal function: Load trained model onto memory 

        Args:
            model_name (str): Select the right model name as per training
            params_file (str): Relative path to params file
            use_gpu (bool): If True use GPU else run on CPU

        Returns:
            None
        '''
        self.system_dict["model_name"] = model_name;
        self.system_dict["params_file"] = params_file;
        self.system_dict["use_gpu"] = use_gpu;

        if(self.system_dict["model_name"] in self.system_dict["model_set_1"]):
            first_name, second_name = self.system_dict["model_name"].split("_");
            if(first_name == "faster-rcnn" and second_name == "mobilenet-v2"):
                backbone = torchvision.models.mobilenet_v2(pretrained=False).features;
                backbone.out_channels = 1280;
                anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),));

                roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2);

                self.system_dict["local"]["model"] = FasterRCNN(backbone,
                                                               num_classes=len(self.system_dict["classes"])+1,
                                                               rpn_anchor_generator=anchor_generator,
                                                               box_roi_pool=roi_pooler);

                self.set_device(use_gpu=self.system_dict["use_gpu"]);

                self.system_dict["local"]["model"].load_state_dict(torch.load(self.system_dict["params_file"]))

                self.system_dict["local"]["model"] = self.system_dict["local"]["model"].eval()


                self.system_dict["local"]["model"].to(self.system_dict["local"]["device"]);





        
    def set_device(self, use_gpu=True):
        '''
        Internal function: Set whether to use GPU or CPU

        Args:
            use_gpu (bool): If True use GPU else run on CPU

        Returns:
            None
        '''
        self.system_dict["use_gpu"] = use_gpu;

        if(self.system_dict["use_gpu"]):
            self.system_dict["local"]["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.system_dict["local"]["device"] = torch.device('cpu');


    def run(self, img_name, visualize=True, thresh=0.9):
        '''
        User function: Run inference on image and visualize it

        Args:
            img_name (str): Relative path to the image file
            visualize (bool): If True, displays image with predicted bounding boxes and scores
            thresh (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 

        Returns:
            dict: Contaning IDs, Scores and bounding box locations of predicted objects. 
        '''
        img = Image.open(img_name);
        tf = self.get_transform(False);
        img, _ = tf(img, None);

        img = img.unsqueeze(0);
        img = Variable(img);

        if(self.system_dict["use_gpu"]):
            img = img.cuda();

        outputs = self.system_dict["local"]["model"](img);

        tmp = {};
        tmp["IDs"] = outputs[0]["labels"];
        tmp["Scores"] = outputs[0]["scores"];
        tmp["Boxes"] = outputs[0]["boxes"];


        img = cv2.imread(img_name);
        for i in range(len(outputs[0]["boxes"])):
            bbox = outputs[0]["boxes"][i];
            bbox = bbox.cpu()
            x1 = int(bbox[0].detach().numpy())
            y1 = int(bbox[1].detach().numpy())
            x2 = int(bbox[2].detach().numpy())
            y2 = int(bbox[3].detach().numpy())
            label = int(outputs[0]["labels"][i])
            
            if(outputs[0]['scores'][i].cpu().detach().numpy() > thresh):
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2);

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (x1,y1)
                fontScale              = 1
                fontColor              = (255,255,255)
                lineType               = 2


                cv2.putText(img, self.system_dict["classes"][label-1], 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imwrite("out.jpg", img);

        return tmp


    def get_transform(self, train):
        '''
        Internal function: Get transforms for training

        Args:
            train (bool):If True, Training tansforms are added, else only test transforms are added

        Returns:
            Torchvision transforms
        '''
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
