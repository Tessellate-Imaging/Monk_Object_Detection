import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
from src.config import colors
import cv2


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
        self.system_dict["local"]["common_size"] = 512;
        self.system_dict["local"]["mean"] = np.array([[[0.485, 0.456, 0.406]]])
        self.system_dict["local"]["std"] = np.array([[[0.229, 0.224, 0.225]]])

    def Model(self, model_dir="trained/"):
        '''
        User function: Selet trained model params

        Args:
            model_dir (str): Relative path to directory containing trained models 

        Returns:
            None
        '''
        self.system_dict["local"]["model"] = torch.load(model_dir + "/signatrix_efficientdet_coco.pth").module
        if torch.cuda.is_available():
            self.system_dict["local"]["model"] = self.system_dict["local"]["model"].cuda();

    def Predict(self, img_path, class_list, vis_threshold = 0.4, output_folder = 'Inference'):
        '''
        User function: Run inference on image and visualize it

        Args:
            img_path (str): Relative path to the image file
            class_list (list): List of classes in the training set
            vis_threshold (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
            output_folder (str): Path to folder where output images will be saved

        Returns:
            tuple: Contaning label IDs, Scores and bounding box locations of predicted objects. 
        '''

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_filename = os.path.basename(img_path)
        img = cv2.imread(img_path);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        image = img.astype(np.float32) / 255.;
        image = (image.astype(np.float32) - self.system_dict["local"]["mean"]) / self.system_dict["local"]["std"]
        height, width, _ = image.shape
        if height > width:
            scale = self.system_dict["local"]["common_size"] / height
            resized_height = self.system_dict["local"]["common_size"]
            resized_width = int(width * scale)
        else:
            scale = self.system_dict["local"]["common_size"] / width
            resized_height = int(height * scale)
            resized_width = self.system_dict["local"]["common_size"]

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((self.system_dict["local"]["common_size"], self.system_dict["local"]["common_size"], 3))
        new_image[0:resized_height, 0:resized_width] = image

        img = torch.from_numpy(new_image)

        with torch.no_grad():
            scores, labels, boxes = self.system_dict["local"]["model"](img.cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale;


        try:
            if boxes.shape[0] > 0:
                output_image = cv2.imread(img_path)

                for box_id in range(boxes.shape[0]):
                    pred_prob = float(scores[box_id])
                    if pred_prob < vis_threshold:
                        break
                    pred_label = int(labels[box_id])
                    xmin, ymin, xmax, ymax = boxes[box_id, :]
                    color = colors[pred_label]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                    text_size = cv2.getTextSize(class_list[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

                    cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                    cv2.putText(
                        output_image, class_list[pred_label] + ' : %.2f' % pred_prob,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)

            cv2.imwrite(os.path.join(output_folder, image_filename), output_image)
            cv2.imwrite("output.jpg", output_image)
            return scores, labels, boxes
        
        except:
            print("NO Object Detected")
            return None

    def predict_batch_of_images(self, img_folder, class_list, vis_threshold = 0.4, output_folder='Inference'):
        '''
        User function: Run inference on multiple images and visualize them

        Args:
            img_folder (str): Relative path to folder containing all the image files
            class_list (list): List of classes in the training set
            vis_threshold (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
            output_folder (str): Path to folder where output images will be saved

        Returns:
            None 
        '''
        all_filenames = os.listdir(img_folder)
        all_filenames.sort()
        generated_count = 0
        for filename in all_filenames:
            img_path = "{}/{}".format(img_folder, filename)
            try:
                self.Predict(img_path , class_list, vis_threshold ,output_folder)
                generated_count += 1
            except:
                continue
        print("Objects detected  for {} images".format(generated_count))

        
