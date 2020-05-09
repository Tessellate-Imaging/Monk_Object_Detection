import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz
import pandas as pd
import cv2
from PIL import Image
from tqdm.notebook import tqdm

from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

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
        self.system_dict["model_set_1"] = ["ssd_300_vgg16_atrous_coco", "ssd_300_vgg16_atrous_voc"];
        self.system_dict["model_set_2"] = ["ssd_512_vgg16_atrous_coco", "ssd_512_vgg16_atrous_voc"];
        self.system_dict["model_set_3"] = ["ssd_512_resnet50_v1_coco", "ssd_512_resnet50_v1_voc"];
        self.system_dict["model_set_4"] = ["ssd_512_mobilenet1.0_voc", "ssd_512_mobilenet1.0_coco"];
        self.system_dict["model_set_5"] = ["yolo3_darknet53_voc", "yolo3_darknet53_coco"];
        self.system_dict["model_set_6"] = ["yolo3_mobilenet1.0_voc", "yolo3_mobilenet1.0_coco"];

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
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model("ssd_300_vgg16_atrous_custom", 
                classes=self.system_dict["classes"], 
                pretrained_base=False);
            self.system_dict["local"]["net"].load_parameters(self.system_dict["params_file"]);
            self.system_dict["img_size"] = (300, 300);

            self.set_device(use_gpu=use_gpu);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif(self.system_dict["model_name"] in self.system_dict["model_set_2"]):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model("ssd_512_vgg16_atrous_custom", 
                classes=self.system_dict["classes"], 
                pretrained_base=False);
            self.system_dict["local"]["net"].load_parameters(self.system_dict["params_file"]);
            self.system_dict["img_size"] = (512, 512);

            self.set_device(use_gpu=use_gpu);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif(self.system_dict["model_name"] in self.system_dict["model_set_3"]):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model("ssd_512_resnet50_v1_custom", 
                classes=self.system_dict["classes"], 
                pretrained_base=False);
            self.system_dict["local"]["net"].load_parameters(self.system_dict["params_file"]);
            self.system_dict["img_size"] = (512, 512);

            self.set_device(use_gpu=use_gpu);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif(self.system_dict["model_name"] in self.system_dict["model_set_4"]):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model("ssd_512_mobilenet1.0_custom", 
                classes=self.system_dict["classes"], 
                pretrained_base=False);
            self.system_dict["local"]["net"].load_parameters(self.system_dict["params_file"]);
            self.system_dict["img_size"] = (512, 512);

            self.set_device(use_gpu=use_gpu);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif(self.system_dict["model_name"] in self.system_dict["model_set_5"]):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model("yolo3_darknet53_custom", 
                classes=self.system_dict["classes"], 
                pretrained_base=False);
            self.system_dict["local"]["net"].load_parameters(self.system_dict["params_file"]);
            self.system_dict["img_size"] = (416, 416);

            self.set_device(use_gpu=use_gpu);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif(self.system_dict["model_name"] in self.system_dict["model_set_6"]):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model("yolo3_mobilenet1.0_custom", 
                classes=self.system_dict["classes"], 
                pretrained_base=False);
            self.system_dict["local"]["net"].load_parameters(self.system_dict["params_file"]);
            self.system_dict["img_size"] = (416, 416);

            self.set_device(use_gpu=use_gpu);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])


    def set_device(self, use_gpu=True):
        '''
        Internal function: Set whether to use GPU or CPU

        Args:
            use_gpu (bool): If True use GPU else run on CPU

        Returns:
            None
        '''
        self.system_dict["use_gpu"] = use_gpu;
        if(use_gpu):
            try:
                a = mx.nd.zeros((1,), ctx=mx.gpu(0))
                self.system_dict["local"]["ctx"] = [mx.gpu(0)]
            except:
                self.system_dict["local"]["ctx"] = [mx.cpu()]
        else:
            self.system_dict["local"]["ctx"] = [mx.cpu()]

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
        x, image = gcv.data.transforms.presets.ssd.load_test(img_name, self.system_dict["img_size"][0])
        self.system_dict["local"]["cid"], self.system_dict["local"]["score"], self.system_dict["local"]["bbox"] = self.system_dict["local"]["net"](x.copyto(self.system_dict["local"]["ctx"][0]))

        tmp = {};
        tmp["IDs"] = self.system_dict["local"]["cid"];
        tmp["Scores"] = self.system_dict["local"]["score"];
        tmp["Boxes"] = self.system_dict["local"]["bbox"];

        ax = viz.plot_bbox(image, self.system_dict["local"]["bbox"][0], self.system_dict["local"]["score"][0], 
                self.system_dict["local"]["cid"][0], class_names=self.system_dict["classes"], thresh=thresh)
        if(visualize):
            plt.show()
        plt.savefig("output.png");
        
        return tmp;



