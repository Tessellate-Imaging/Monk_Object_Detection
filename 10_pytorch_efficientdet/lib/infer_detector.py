import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};


        self.system_dict["params"] = {};

        self.system_dict["params"]["weights_file"] = "";
        self.system_dict["params"]["obj_list"] = [];
        self.system_dict["params"]["use_cuda"] = True;
        self.system_dict["params"]["threshold"] = 0.5
        self.system_dict["params"]["iou_threshold"] = 0.2
        self.system_dict["params"]["img_path"] = "";
                                   
                                   
        
        self.system_dict["params"]["force_input_size"] = None;
        self.system_dict["params"]["anchor_ratios"] = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)];
        self.system_dict["params"]["anchor_scales"] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)];
        self.system_dict["params"]["use_float16"] = False;
        self.system_dict["params"]["cudnn.fastest"] = True;
        self.system_dict["params"]["cudnn.benchmark"] = True;
        
    
    
    def load_model(self, model_path, classes_list, use_gpu=True):
        if("d0" in model_path):
            self.system_dict["params"]["compound_coef"] = 0;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d1" in model_path):
            self.system_dict["params"]["compound_coef"] = 1;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d2" in model_path):
            self.system_dict["params"]["compound_coef"] = 2;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d3" in model_path):
            self.system_dict["params"]["compound_coef"] = 3;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d4" in model_path):
            self.system_dict["params"]["compound_coef"] = 4;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d5" in model_path):
            self.system_dict["params"]["compound_coef"] = 5;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d6" in model_path):
            self.system_dict["params"]["compound_coef"] = 6;
            self.system_dict["params"]["weights_file"] = model_path;
        elif("d7" in model_path):
            self.system_dict["params"]["compound_coef"] = 7;
            self.system_dict["params"]["weights_file"] = model_path;
        
        self.system_dict["params"]["obj_list"] = classes_list;
        self.system_dict["params"]["use_cuda"] = use_gpu
        
        
        self.system_dict["local"]["color_list"] = standard_to_bgr(STANDARD_COLORS)
        
        # tf bilinear interpolation is different from any other's, just make do
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.system_dict["local"]["input_size"] = input_sizes[self.system_dict["params"]["compound_coef"]] if self.system_dict["params"]["force_input_size"] is None else self.system_dict["params"]["force_input_size"]
        
        self.system_dict["local"]["model"] = EfficientDetBackbone(compound_coef=self.system_dict["params"]["compound_coef"], 
                                             num_classes=len(self.system_dict["params"]["obj_list"]),
                                             ratios=self.system_dict["params"]["anchor_ratios"], 
                                             scales=self.system_dict["params"]["anchor_scales"])
        
        self.system_dict["local"]["model"].load_state_dict(torch.load(self.system_dict["params"]["weights_file"]))
        self.system_dict["local"]["model"].requires_grad_(False)
        self.system_dict["local"]["model"] = self.system_dict["local"]["model"].eval()

        if self.system_dict["params"]["use_cuda"]:
            self.system_dict["local"]["model"] = self.system_dict["local"]["model"].cuda()
        if self.system_dict["params"]["use_float16"]:
            self.system_dict["local"]["model"] = model.half()
        
        
    def predict(self, img_path, threshold=0.5):
        self.system_dict["params"]["threshold"] = threshold;
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, 
                                                         max_size=self.system_dict["local"]["input_size"])

        if self.system_dict["params"]["use_cuda"]:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not self.system_dict["params"]["use_float16"] else torch.float16).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            features, regression, classification, anchors = self.system_dict["local"]["model"](x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              self.system_dict["params"]["threshold"], self.system_dict["params"]["iou_threshold"])
            
        out = invert_affine(framed_metas, out)
        scores, labels, bboxes = self.display(out, ori_imgs, imshow=False, imwrite=True)
        return scores, labels, bboxes;    
     
    
    
    def display(self, preds, imgs, imshow=True, imwrite=False):
        scores = [];
        labels = [];
        bboxes = [];
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = self.system_dict["params"]["obj_list"][preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                if(score > self.system_dict["params"]["threshold"]):
                    scores.append(score);
                    labels.append(obj);
                    bboxes.append([x1, y1, x2, y2]);
                    plot_one_box(imgs[i], 
                                 [x1, y1, x2, y2], 
                                 label=obj,
                                 score=score,
                                 color=self.system_dict["local"]["color_list"][get_index_label(obj, self.system_dict["params"]["obj_list"])])


            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
                cv2.imwrite('output.jpg', imgs[i])
                 
        return scores, labels, bboxes;                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
