import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import shutil
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models as sm
from utils import Dataset
from utils import Dataloder
from utils import get_training_augmentation
from utils import get_validation_augmentation
from utils import get_preprocessing
from utils import visualize
from utils import visualize2
from utils import denormalize


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


    def Data_Params(self, classes_dict, classes_to_train, image_shape=[320, 320]):
        '''
        User function: Set dataset parameters

        Args:
            classes_dict (dict): Dictionary of classes with integer labels
            classes_to_train (list): List of classes to be trained from all available classes
                                     (If None, all the classes will be trained)


        Returns:
            None
        '''
        self.system_dict["params"]["classes_dict"] = classes_dict;
        self.system_dict["params"]["classes_to_train"] = classes_to_train;
        self.system_dict["params"]["image_shape"] = image_shape;


    def Model_Params(self, model="Unet", backbone="efficientnetb2", path_to_model='best_model.h5'):
        '''
        User function: Set Model parameters

            Available Models
                Unet
                FPN
                Linknet
                PSPNet

            Available backbones
                'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                'resnext50', 'resnext101', 'seresnext50', 'seresnext101', 'senet154',
                'densenet121', 'densenet169', 'densenet201', 'inceptionv3', 'inceptionresnetv2',
                'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 
                'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'

        Args:
            model (str): Select model from available models
            backbone (str): Select backbone network for feature extraction
            path_to_model (str): Relative path to the trained model 


        Returns:
            None
        '''
        self.system_dict["params"]["model"] = model;
        self.system_dict["params"]["backbone"] = backbone;
        self.system_dict["params"]["path_to_model"] = path_to_model;


    def Setup(self):
        '''
        User function: Setup all the parameters

        Args:
            None

        Returns:
            None
        '''
        # define network parameters
        self.system_dict["local"]["n_classes"] = 1 if len(self.system_dict["params"]["classes_to_train"]) == 1 else (len(self.system_dict["params"]["classes_to_train"]) + 1)  # case for binary and multiclass segmentation
        activation = 'sigmoid' if self.system_dict["local"]["n_classes"] == 1 else 'softmax'

        #create model
        if(self.system_dict["params"]["model"] == "Unet"):
            self.system_dict["local"]["model"] = sm.Unet(self.system_dict["params"]["backbone"], 
                                                            classes=self.system_dict["local"]["n_classes"], activation=activation)
        elif(self.system_dict["params"]["model"] == "FPN"):
            self.system_dict["local"]["model"] = sm.FPN(self.system_dict["params"]["backbone"], 
                                                            classes=self.system_dict["local"]["n_classes"], activation=activation)
        elif(self.system_dict["params"]["model"] == "Linknet"):
            self.system_dict["local"]["model"] = sm.Linknet(self.system_dict["params"]["backbone"], 
                                                            classes=self.system_dict["local"]["n_classes"], activation=activation)
        elif(self.system_dict["params"]["model"] == "PSPNet"):
            self.system_dict["local"]["model"] = sm.PSPNet(self.system_dict["params"]["backbone"], 
                                                            classes=self.system_dict["local"]["n_classes"], activation=activation)



        # define optomizer
        optim = keras.optimizers.Adam(0.0001)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if self.system_dict["local"]["n_classes"] == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimozer, loss and metrics
        self.system_dict["local"]["model"].compile(optim, total_loss, metrics)

        self.system_dict["local"]["model"].load_weights(self.system_dict["params"]["path_to_model"]) 



    def Predict(self, img_path, vis=True):
        '''
        User function: Run inference on image and visualize it. Output mask saved as output_mask.npy

        Args:
            img_path (str): Relative path to the image file
            vis (bool): If True, predicted mask is displayed.

        Returns:
            list: List of bounding box locations of predicted objects along with classes. 
        '''
        dirPath = "tmp_test"
        
        if(os.path.isdir(dirPath)):
            shutil.rmtree(dirPath);

        os.mkdir(dirPath);
        os.mkdir(dirPath + "/img_dir");
        os.mkdir(dirPath + "/gt_dir");
            
        os.system("cp " + img_path + " " + dirPath + "/img_dir")
        os.system("cp " + img_path + " " + dirPath + "/gt_dir")


        x_test_dir = dirPath + "/img_dir";
        y_test_dir = dirPath + "/gt_dir";

        if(self.system_dict["params"]["image_shape"][0] % 32 != 0):
            self.system_dict["params"]["image_shape"][0] += (32 - self.system_dict["params"]["image_shape"][0] % 32)

        if(self.system_dict["params"]["image_shape"][1] % 32 != 0):
            self.system_dict["params"]["image_shape"][1] += (32 - self.system_dict["params"]["image_shape"][1] % 32)

        preprocess_input = sm.get_preprocessing(self.system_dict["params"]["backbone"])
        test_dataset = Dataset(
            x_test_dir, 
            y_test_dir, 
            self.system_dict["params"]["classes_dict"],
            classes_to_train=self.system_dict["params"]["classes_to_train"], 
            augmentation=get_validation_augmentation(self.system_dict["params"]["image_shape"][0], self.system_dict["params"]["image_shape"][1]),
            preprocessing=get_preprocessing(preprocess_input),
        )

        test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

        image, gt_mask = test_dataset[0]
        image = np.expand_dims(image, axis=0)
        pr_mask = self.system_dict["local"]["model"].predict(image).round()
        np.save("output_mask.npy", pr_mask)

        '''
        if(vis):
            visualize(
                image=denormalize(image.squeeze()),
                pr_mask=pr_mask[..., 0].squeeze(),
            )
        '''
        
        img_list = [denormalize(image.squeeze())];
        label_list = ["image"];

        
        for i in range(len(self.system_dict["params"]["classes_to_train"])):
            img_list.append(pr_mask[..., i].squeeze());
            label_list.append(self.system_dict["params"]["classes_to_train"][i])
            
        
        
        if(vis):
            visualize2(
                label_list,
                img_list
            )
