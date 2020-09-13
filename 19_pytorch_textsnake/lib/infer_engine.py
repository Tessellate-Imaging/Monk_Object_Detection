import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset.deploy import DeployDataset
from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.visualize import visualize_detection
from util.misc import to_device, mkdirs, rescale_result

from dataset.data_util import pil_load_img
from util.augmentation import BaseTransform, Augmentation
from torch.utils.data import DataLoader


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
        
    def Dataset_Params(self, input_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.system_dict["local"]["input_size"] = input_size;
        self.system_dict["local"]["means"] = mean;
        self.system_dict["local"]["stds"] = std;
        
    def Model_Params(self, model_type="vgg", model_path=None, use_gpu=True):
        self.system_dict["local"]["net"] = model_type;
        self.system_dict["local"]["model_path"] = model_path;
        self.system_dict["local"]["cuda"] = use_gpu;
        
        self.system_dict["local"]["cfg"] = cfg;
        
        self.system_dict["local"]["cfg"].net = self.system_dict["local"]["net"];
        self.system_dict["local"]["cfg"].cuda = self.system_dict["local"]["cuda"];
        self.system_dict["local"]["cfg"].means = self.system_dict["local"]["means"];
        self.system_dict["local"]["cfg"].stds = self.system_dict["local"]["stds"];
        self.system_dict["local"]["cfg"].input_size = self.system_dict["local"]["input_size"];
        
        model = TextNet(is_training=False, backbone=self.system_dict["local"]["cfg"].net)
        model.load_model(model_path)

        # copy to cuda
        if(self.system_dict["local"]["cfg"].cuda):
            cudnn.benchmark = True
            self.system_dict["local"]["cfg"].device = torch.device("cuda")
        else:
            self.system_dict["local"]["cfg"].device = torch.device("cpu")


        self.system_dict["local"]["model"] = model.to(self.system_dict["local"]["cfg"].device)  
        
    def write_to_file(self, contours, file_path):
        """
        :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
        :param file_path: target file path
        """
        # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
        with open(file_path, 'w') as f:
            for cont in contours:
                cont = np.stack([cont[:, 1], cont[:, 0]], 1)
                cont = cont.flatten().astype(str).tolist()
                cont = ','.join(cont)
                f.write(cont + '\n')

    def Predict(self, 
                image_path, 
                output_img_path="output.jpg",
                output_txt_path="output.txt",
                tr_thresh = 0.4,
                tcl_thresh = 0.4):
        
        cfg = self.system_dict["local"]["cfg"];
        model = self.system_dict["local"]["model"];
        
        start = time.time();
        image = pil_load_img(image_path)

        transform = BaseTransform(
                size=cfg.input_size, mean=cfg.means, std=cfg.stds
            )

        H, W, _ = image.shape

        image, polygons = transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': 0,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        image = torch.from_numpy(np.expand_dims(image, axis=0))
        image = to_device(image)
        if(self.system_dict["local"]["cfg"].cuda):
            torch.cuda.synchronize()
        
        end = time.time();
        print("Image loading time: {}".format(end-start));
        
        
        
        start = time.time()
        detector = TextDetector(model, tr_thresh=tr_thresh, tcl_thresh=tcl_thresh)
        # get detection result
        contours, output = detector.detect(image)

        torch.cuda.synchronize()
        end = time.time()

        print("Inference time - {}".format(end-start))
        
        
        start = time.time();
        tr_pred, tcl_pred = output['tr'], output['tcl']
        img_show = image[0].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        img_show, contours = rescale_result(img_show, contours, H, W)

        pred_vis = visualize_detection(img_show, contours)
        cv2.imwrite(output_img_path, pred_vis)

        # write to file
        self.write_to_file(contours, output_txt_path)
        end = time.time()

        print("Writing output time - {}".format(end-start))














