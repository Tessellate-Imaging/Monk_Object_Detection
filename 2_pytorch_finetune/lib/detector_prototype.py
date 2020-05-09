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

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
if(isnotebook()):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm as tqdm


import transforms as T
from data_loader_class import CustomDatasetMultiObject

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class Detector():
    '''
    Class to train a detector

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["model_set_1"] = ["faster-rcnn_mobilenet-v2"]



    def Dataset(self, train_dataset, val_dataset=None, batch_size=4, num_workers=4):
        '''
        User function: Set dataset parameters

        Dataset Directory Structure

            Parent_Directory (root)
                    |   
                    |-----------Images (img_dir)
                    |              |
                    |              |------------------img1.jpg
                    |              |------------------img2.jpg
                    |              |------------------.........(and so on)
                    |
                    |
                    |-----------train_labels.csv (anno_file)

        Annotation file format

               | Id         | Labels                                 |
               | img1.jpg   | x1 y1 x2 y2 label1 x1 y1 x2 y2 label2  |

            Labels: xmin ymin xmax ymax label
            xmin, ymin - top left corner of bounding box
            xmax, ymax - bottom right corner of bounding box


        Args:
            train_dataset (list): 
                First element - Path to root folder containing training images and label file\n
                Second element - Name of directory containing images\n
                Third element - Name of Labels files containing annotations in monk format
            val_dataset (list): Optional \n
                First element - Path to root folder containing training images and label file\n 
                Second element - Name of directory containing images\n
                Third element - Name of Labels files containing annotations in monk format
            batch_size (int): Mini batch sampling size for training epochs
            num_workers (int): Number of parallel processors for data loader 

        Returns:
            None
        '''
        train_root = train_dataset[0];
        train_img_dir = train_dataset[1];
        train_anno_file = train_dataset[2];

        self.system_dict["train_root"] = train_root;
        self.system_dict["train_img_dir"] = train_img_dir;
        self.system_dict["train_anno_file"] = train_anno_file;

        if(val_dataset):
            val_root = val_dataset[0];
            val_img_dir = val_dataset[1];
            val_anno_file = val_dataset[2];

            self.system_dict["val_root"] = val_root;
            self.system_dict["val_img_dir"] = val_img_dir;
            self.system_dict["val_anno_file"] = val_anno_file;

        else:
            self.system_dict["val_root"] = train_root;
            self.system_dict["val_img_dir"] = train_img_dir;
            self.system_dict["val_anno_file"] = train_anno_file;


        self.system_dict["batch_size"] = batch_size;
        self.system_dict["num_workers"] = num_workers;

        self.system_dict["local"]["train_list"] = pd.read_csv(self.system_dict["train_root"] + 
            self.system_dict["train_anno_file"]);
        columns = self.system_dict["local"]["train_list"].columns;
        classes = [];
        for i in tqdm(range(len(self.system_dict["local"]["train_list"]))):
            tmp = self.system_dict["local"]["train_list"]["Label"][i].split(" ");
            for j in range(len(tmp)//5):
                label = tmp[j*5+4];
                if(label not in classes):
                    classes.append(label)
        self.system_dict["classes"] = sorted(classes)


        self.system_dict["local"]["dataset"] = CustomDatasetMultiObject(self.system_dict["train_root"], 
                                            self.system_dict["train_img_dir"],
                                            self.system_dict["train_anno_file"],
                                              self.get_transform(train=True))
        self.system_dict["local"]["dataset_test"] = CustomDatasetMultiObject(self.system_dict["val_root"], 
                                                self.system_dict["val_img_dir"],
                                                self.system_dict["val_anno_file"],
                                                self.get_transform(train=False))
        
        self.system_dict["local"]["num_classes"] = self.system_dict["local"]["dataset"].num_classes;


        # split the dataset in train and test set
        indices = torch.randperm(len(self.system_dict["local"]["dataset"])).tolist()
        self.system_dict["local"]["dataset"] = torch.utils.data.Subset(self.system_dict["local"]["dataset"], indices[:])
        self.system_dict["local"]["dataset_test"] = torch.utils.data.Subset(self.system_dict["local"]["dataset_test"], indices[:])


        # define training and validation data loaders
        self.system_dict["local"]["data_loader"] = torch.utils.data.DataLoader(
            self.system_dict["local"]["dataset"], batch_size=self.system_dict["batch_size"], shuffle=True, 
            num_workers=self.system_dict["num_workers"],
            collate_fn=utils.collate_fn)

        self.system_dict["local"]["data_loader_test"] = torch.utils.data.DataLoader(
            self.system_dict["local"]["dataset_test"], batch_size=self.system_dict["batch_size"], shuffle=False, 
            num_workers=self.system_dict["num_workers"],
            collate_fn=utils.collate_fn)






    def Model(self, model_name, use_pretrained=True, use_gpu=True):
        '''
        User function: Set Model parameters

            Available models
                faster-rcnn_mobilenet-v2

        Args:
            model_name (str): Select from available models
            use_pretrained (bool): If True use pretrained weights else randomly initialized weights
            use_gpu (bool): If True use GPU else run on CPU

        Returns:
            None
        '''

        self.system_dict["model_name"] = model_name;
        self.system_dict["use_pretrained"] = use_pretrained;
        self.system_dict["use_gpu"] = use_gpu;

        if(self.system_dict["model_name"] in self.system_dict["model_set_1"]):
            first_name, second_name = self.system_dict["model_name"].split("_");
            if(first_name == "faster-rcnn" and second_name == "mobilenet-v2"):
                backbone = torchvision.models.mobilenet_v2(pretrained=use_pretrained).features;
                backbone.out_channels = 1280;
                anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),));

                roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2);

                self.system_dict["local"]["model"] = FasterRCNN(backbone,
                                                               num_classes=self.system_dict["local"]["num_classes"],
                                                               rpn_anchor_generator=anchor_generator,
                                                               box_roi_pool=roi_pooler);

                self.set_device(use_gpu=self.system_dict["use_gpu"]);


                self.system_dict["local"]["model"].to(self.system_dict["local"]["device"]);



    def set_device(self, use_gpu=True):
        '''
        Internal function: Prepares GPU and CPU devices as per the model parameters set

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


    def Set_Learning_Rate(self, lr):
        '''
        User function: Set initial learning rate

        Args:
            lr (float): Base learning rate

        Returns:
            None
        '''
        params = [p for p in  self.system_dict["local"]["model"].parameters() if p.requires_grad]
        self.system_dict["local"]["optimizer"] = torch.optim.SGD(params, lr=lr,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        self.system_dict["local"]["lr_scheduler"] = torch.optim.lr_scheduler.StepLR(self.system_dict["local"]["optimizer"],
                                                                                   step_size=5,
                                                                                   gamma=0.1)


    def Train(self, epochs, params_file):
        '''
        User function: Start training

        Args:
            epochs (int): Number of epochs to train for
            params_file (str): Trained weights file name with extension as ".params"

        Returns:
            None
        '''
        self.system_dict["num_epochs"] = epochs;
        self.system_dict["params_file"] = params_file;

        for epoch in range(self.system_dict["num_epochs"]):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.system_dict["local"]["model"], 
                            self.system_dict["local"]["optimizer"], 
                            self.system_dict["local"]["data_loader"], 
                            self.system_dict["local"]["device"], 
                            epoch, print_freq=20)
            # update the learning rate
            self.system_dict["local"]["lr_scheduler"].step()
            # evaluate on the test dataset
            evaluate(self.system_dict["local"]["model"], 
                    self.system_dict["local"]["data_loader_test"], 
                    device=self.system_dict["local"]["device"]);

        torch.save(self.system_dict["local"]["model"].state_dict(), params_file)
        



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

        

