import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


class Segmenter():
    '''
    Class to train a segmentation algorithm

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};

        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;

        self.system_dict["params"] = {};
        self.system_dict["params"]["backbones"] = ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                    'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                                                    'resnext50', 'resnext101', 'seresnext50', 'seresnext101', 'senet154',
                                                    'densenet121', 'densenet169', 'densenet201', 'inceptionv3', 'inceptionresnetv2',
                                                    'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 
                                                    'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'
                                                    ];
        self.system_dict["params"]["models"] = ["Unet", "FPN", "Linknet", "PSPNet"];
        
        self.system_dict["params"]["callbacks"] = [
                        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
                        keras.callbacks.ReduceLROnPlateau(),
                    ]



    def Train_Dataset(self, img_dir, mask_dir, classes_dict, classes_to_train=None):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                       root_dir
                          |
                          | 
                          |         
                          |----train_img_dir
                          |       |
                          |       |---------img1.jpg
                          |       |---------img2.jpg
                          |                |---------..........(and so on) 
                          |
                          |----train_mask_dir
                          |       |
                          |       |---------img1.jpg
                          |       |---------img2.jpg
                          |                |---------..........(and so on)

        Args:
            img_dir (str): Path to directory containing images
            mask_dir (str): Path to directory containing label masks
            classes_dict (dict): Dictionary of classes with integer labels
            classes_to_train (list): List of classes to be trained from all available classes
                                     (If None, all the classes will be trained)


        Returns:
            None
        '''
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["mask_dir"] = mask_dir;
        self.system_dict["dataset"]["train"]["classes_dict"] = classes_dict;
        self.system_dict["dataset"]["train"]["classes_to_train"] = classes_to_train;


    def Val_Dataset(self, img_dir, mask_dir):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                       root_dir
                          |
                          |
                          |----val_img_dir 
                          |       |
                          |       |---------img1.jpg
                          |       |---------img2.jpg
                          |                |---------..........(and so on)
                          |
                          |----val_mask_dir
                          |       |
                          |       |---------img1.jpg
                          |       |---------img2.jpg
                          |                |---------..........(and so on)

        Args: Optional \n
            img_dir (str): Path to directory containing images
            mask_dir (str): Path to directory containing label masks

        Returns:
            None
        '''
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["mask_dir"] = mask_dir;
        self.system_dict["dataset"]["val"]["status"] = True;


    def List_Backbones(self):
        '''
        User function: List all backbone

        Args:
            None

        Returns:
            None
        '''
        print("Available backbones - {}".format(self.system_dict["params"]["backbones"]));


    def List_Models(self):
        '''
        User function: List all models

        Args:
            None

        Returns:
            None
        '''
        print("Available models - {}".format(self.system_dict["params"]["models"]));


    def Data_Params(self, batch_size=2, backbone="efficientnetb2", image_shape=(320, 320)):
        '''
        User function: Set Data parameters

            Available backbones
                'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                'resnext50', 'resnext101', 'seresnext50', 'seresnext101', 'senet154',
                'densenet121', 'densenet169', 'densenet201', 'inceptionv3', 'inceptionresnetv2',
                'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 
                'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'

        Args:
            batch_size (int):  Mini batch sampling size for training epochs
            backbone (str): Select backbone network for feature extraction

        Returns:
            None
        '''
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["backbone"] = backbone;
        self.system_dict["params"]["image_shape"] = image_shape;


    def Model_Params(self, model="Unet"):
        '''
        User function: Set Model parameters

            Available Models
                Unet
                FPN
                Linknet
                PSPNet

        Args:
            model (str): Select model from available models

        Returns:
            None
        '''
        self.system_dict["params"]["model"] = model;


    def Train_Params(self, lr=0.0001):
        '''
        User function: Set Training parameters

        Args:
            lr (float): Initial learning rate for training

        Returns:
            None
        '''
        self.system_dict["params"]["lr"] = lr;

    
    def Setup(self):
        '''
        User function: Setup all the parameters

        Args:
            None

        Returns:
            None
        '''
        preprocess_input = sm.get_preprocessing(self.system_dict["params"]["backbone"])
        # define network parameters
        self.system_dict["local"]["n_classes"] = 1 if len(self.system_dict["dataset"]["train"]["classes_to_train"]) == 1 else (len(self.system_dict["dataset"]["train"]["classes_to_train"]) + 1)  # case for binary and multiclass segmentation
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
        optim = keras.optimizers.Adam(self.system_dict["params"]["lr"])

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if self.system_dict["local"]["n_classes"] == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimozer, loss and metrics
        self.system_dict["local"]["model"].compile(optim, total_loss, metrics)
        


        # Dataset for train images
        train_dataset = Dataset(
            self.system_dict["dataset"]["train"]["img_dir"], 
            self.system_dict["dataset"]["train"]["mask_dir"], 
            self.system_dict["dataset"]["train"]["classes_dict"],
            classes_to_train=self.system_dict["dataset"]["train"]["classes_to_train"], 
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )


        if(self.system_dict["params"]["image_shape"][0] % 32 != 0):
            self.system_dict["params"]["image_shape"][0] += (32 - self.system_dict["params"]["image_shape"][0] % 32)

        if(self.system_dict["params"]["image_shape"][1] % 32 != 0):
            self.system_dict["params"]["image_shape"][1] += (32 - self.system_dict["params"]["image_shape"][1] % 32)



        # Dataset for validation images
        if(self.system_dict["dataset"]["val"]["status"]):
            valid_dataset = Dataset(
                self.system_dict["dataset"]["val"]["img_dir"], 
                self.system_dict["dataset"]["val"]["mask_dir"], 
                self.system_dict["dataset"]["train"]["classes_dict"],
                classes_to_train=self.system_dict["dataset"]["train"]["classes_to_train"], 
                augmentation=get_validation_augmentation(self.system_dict["params"]["image_shape"][0], self.system_dict["params"]["image_shape"][1]),
                preprocessing=get_preprocessing(preprocess_input),
            )
        else:
            valid_dataset = Dataset(
                self.system_dict["dataset"]["train"]["img_dir"], 
                self.system_dict["dataset"]["train"]["mask_dir"], 
                self.system_dict["dataset"]["train"]["classes_dict"],
                classes_to_train=self.system_dict["dataset"]["train"]["classes_to_train"], 
                augmentation=get_validation_augmentation(self.system_dict["params"]["image_shape"][0], self.system_dict["params"]["image_shape"][1]),
                preprocessing=get_preprocessing(preprocess_input),
            )


        self.system_dict["local"]["train_dataloader"] = Dataloder(train_dataset, 
                                                                    batch_size=self.system_dict["params"]["batch_size"], 
                                                                    shuffle=True)
        self.system_dict["local"]["valid_dataloader"] = Dataloder(valid_dataset, 
                                                                    batch_size=1, 
                                                                    shuffle=False)





    def Train(self, num_epochs=2):
        '''
        User function: Start training

        Args:
            num_epochs (int): Number of epochs to train for

        Returns:
            None
        '''
        # train model
        self.system_dict["local"]["history"] = self.system_dict["local"]["model"].fit_generator(
            self.system_dict["local"]["train_dataloader"], 
            steps_per_epoch=len(self.system_dict["local"]["train_dataloader"]), 
            epochs=num_epochs, 
            callbacks=self.system_dict["params"]["callbacks"], 
            validation_data=self.system_dict["local"]["valid_dataloader"], 
            validation_steps=len(self.system_dict["local"]["valid_dataloader"]),
        )


    def Visualize_Training_History(self):
        '''
        User function: Plot training and validation history

        Args:
            num_epochs (int): Number of epochs to train for

        Returns:
            None
        '''
        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(self.system_dict["local"]["history"].history['iou_score'])
        plt.plot(self.system_dict["local"]["history"].history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(self.system_dict["local"]["history"].history['loss'])
        plt.plot(self.system_dict["local"]["history"].history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


