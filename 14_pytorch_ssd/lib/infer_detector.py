from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import time
import torch

class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["params"] = {};
        
    
    def load_model(self, net="mb1-ssd", model_path=None, label_path=None, use_gpu=True):
        net_type=net;
        self.system_dict["device"] = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.system_dict["class_names"] = [name.strip() for name in open(label_path).readlines()]
        self.system_dict["class_names"].insert(0, 'BACKGROUND');
        
        if net_type == 'vgg16-ssd':
            self.system_dict["net"] = create_vgg_ssd(len(self.system_dict["class_names"]), 
                                                     is_test=True, device=self.system_dict["device"])
        elif net_type == 'mb1-ssd':
            self.system_dict["net"] = create_mobilenetv1_ssd(len(self.system_dict["class_names"]), 
                                                             is_test=True, device=self.system_dict["device"])
        elif net_type == 'mb1-ssd-lite':
            self.system_dict["net"] = create_mobilenetv1_ssd_lite(len(self.system_dict["class_names"]), 
                                                                  is_test=True, device=self.system_dict["device"])
        elif net_type == 'mb2-ssd-lite':
            self.system_dict["net"] = create_mobilenetv2_ssd_lite(len(self.system_dict["class_names"]), 
                                                                  is_test=True, device=self.system_dict["device"])
        elif net_type == 'sq-ssd-lite':
            self.system_dict["net"] = create_squeezenet_ssd_lite(len(self.system_dict["class_names"]), 
                                                                 is_test=True, device=self.system_dict["device"])
        else:
            print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            sys.exit(1)
        self.system_dict["net"].load(model_path)
        
        
        if net_type == 'vgg16-ssd':
            self.system_dict["predictor"] = create_vgg_ssd_predictor(self.system_dict["net"], 
                                                 candidate_size=200, device=self.system_dict["device"])
        elif net_type == 'mb1-ssd':
            self.system_dict["predictor"] = create_mobilenetv1_ssd_predictor(self.system_dict["net"], 
                                                         candidate_size=200, device=self.system_dict["device"])
        elif net_type == 'mb1-ssd-lite':
            self.system_dict["predictor"] = create_mobilenetv1_ssd_lite_predictor(self.system_dict["net"], 
                                                              candidate_size=200, device=self.system_dict["device"])
        elif net_type == 'mb2-ssd-lite':
            self.system_dict["predictor"] = create_mobilenetv2_ssd_lite_predictor(self.system_dict["net"], 
                                                              candidate_size=200, device=self.system_dict["device"])
        elif net_type == 'sq-ssd-lite':
            self.system_dict["predictor"] = create_squeezenet_ssd_lite_predictor(self.system_dict["net"], 
                                                             candidate_size=200, device=self.system_dict["device"])
        else:
            self.system_dict["predictor"] = create_vgg_ssd_predictor(self.system_dict["net"], 
                                                 candidate_size=200, device=self.system_dict["device"])
            
        
        
    def predict(self, image_path, thresh=0.5, font_scale=1, line_width=3, color=(0, 0, 255)):
        start = time.time();
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        end = time.time();
        print("Image loading time = ", end - start);

        start = time.time();
        boxes, labels, probs = self.system_dict["predictor"].predict(image, 10, thresh)
        end = time.time();
        print("Inference time = ", end - start);

        class_names = self.system_dict["class_names"]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, line_width)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,  # font scale
                        color,
                        2)  # line type
        path = "output.jpg"
        cv2.imwrite(path, orig_image)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        