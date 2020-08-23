import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


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
        self.system_dict["params"] = {};
        self.system_dict["params"]["val_dataset"] = False;
        
    def Train_Dataset(self, img_folder, anno_file, class_file):
        self.system_dict["params"]["train_img_folder"] = img_folder;
        self.system_dict["params"]["train_anno_file"] = anno_file;
        f = open(class_file);
        lines = f.readlines();
        f.close();
        classes = [];
        for i in range(len(lines)):
            if(lines[i] != ""):
                classes.append(lines[i][:len(lines[i])-1]);
        self.system_dict["params"]["classes"]  = tuple(classes);
        
        #Temporary
        self.system_dict["params"]["val_img_folder"] = img_folder;
        self.system_dict["params"]["val_anno_file"] = anno_file;
        
        
    def Val_Dataset(self, img_folder, anno_file):
        self.system_dict["params"]["val_img_folder"] = img_folder;
        self.system_dict["params"]["val_anno_file"] = anno_file;
        self.system_dict["params"]["val_dataset"] = True;
        
    def Dataset_Params(self, batch_size=2, num_workers=2):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
        
    def List_Models(self):
        self.system_dict["params"]["model_list"] = ["faster_rcnn_fpn50", "faster_rcnn_fpn101", 
                                                    "faster_rcnn_x101_32x4d_fpn", "faster_rcnn_x101_64x4d_fpn",
                                                    "cascade_rcnn_fpn50", "cascade_rcnn_fpn101", 
                                                    "cascade_rcnn_x101_32x4d_fpn", "cascade_rcnn_x101_64x4d_fpn",
                                                    "retinanet_r50_fpn", "retinanet_r101_fpn",
                                                    "retinanet_x101_32x4d_fpn", "retinanet_x101_64x4d_fpn",
                                                    "retinanet_ghm_r50_fpn", "retinanet_ghm_r101_fpn",
                                                    "retinanet_ghm_x101_32x4d_fpn", "retinanet_ghm_x101_64x4d_fpn",
                                                    "dh_faster_rcnn_fpn50", "libra_faster_rcnn_fpn50", 
                                                    "libra_faster_rcnn_fpn101", "libra_faster_rcnn_x101_64x4d_fpn",
                                                    "libra_retinanet_r50_fpn", "ga_faster_rcnn_x101_32x4d_fpn",
                                                    "ga_faster_rcnn_x101_64x4d_fpn", "ga_retinanet_x101_32x4d_fpn",
                                                    "ga_retinanet_x101_64x4d_fpn", "fovea_r50_fpn_4x4",
                                                    "fovea_r101_fpn_4x4", "fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4",
                                                    "fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4", 
                                                    "free_anchor_retinanet_r50_fpn", "free_anchor_retinanet_r101_fpn",
                                                    "free_anchor_retinanet_x101_32x4d_fpn", "atss_r50_fpn"]
        
        for i in range(len(self.system_dict["params"]["model_list"])):
            print("{}. Model - {}".format(i+1, self.system_dict["params"]["model_list"][i]));
            
            
            
    def Model_Params(self, model_name="faster_rcnn_fpn50", gpu_devices=[0]):
        self.system_dict["params"]["gpus"] = len(gpu_devices);
        self.system_dict["params"]["gpu_ids"] = gpu_devices;
        if(model_name == "faster_rcnn_fpn50"):
            self.system_dict["local"]["model_name"] = "faster_rcnn_fpn50_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth";
        elif(model_name == "faster_rcnn_fpn101"):
            self.system_dict["local"]["model_name"] = "faster_rcnn_fpn101_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth";
        elif(model_name == "faster_rcnn_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "faster_rcnn_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth";
        elif(model_name == "faster_rcnn_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "faster_rcnn_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth";
        elif(model_name == "cascade_rcnn_fpn50"):
            self.system_dict["local"]["model_name"] = "cascade_rcnn_fpn50_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth";  
        elif(model_name == "cascade_rcnn_fpn101"):
            self.system_dict["local"]["model_name"] = "cascade_rcnn_fpn101_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth"; 
        elif(model_name == "cascade_rcnn_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "cascade_rcnn_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth"; 
        elif(model_name == "cascade_rcnn_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "cascade_rcnn_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth"; 
        elif(model_name == "retinanet_r50_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_r50_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"; 
        elif(model_name == "retinanet_r101_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_r101_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"; 
        elif(model_name == "retinanet_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth";
        elif(model_name == "retinanet_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth";
        elif(model_name == "retinanet_ghm_r50_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_ghm_r50_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth";
        elif(model_name == "retinanet_ghm_r101_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_ghm_r101_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth";
        elif(model_name == "retinanet_ghm_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_ghm_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth";
        elif(model_name == "retinanet_ghm_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_ghm_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth";
        elif(model_name == "dh_faster_rcnn_fpn50"):
            self.system_dict["local"]["model_name"] = "dh_faster_rcnn_fpn50_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth";
        elif(model_name == "libra_faster_rcnn_fpn50"):
            self.system_dict["local"]["model_name"] = "libra_faster_rcnn_fpn50_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth";
        elif(model_name == "libra_faster_rcnn_fpn101"):
            self.system_dict["local"]["model_name"] = "libra_faster_rcnn_fpn101_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth";
        elif(model_name == "libra_faster_rcnn_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "libra_faster_rcnn_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth";
        elif(model_name == "libra_retinanet_r50_fpn"):
            self.system_dict["local"]["model_name"] = "libra_retinanet_r50_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth";
        elif(model_name == "ga_faster_rcnn_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "ga_faster_rcnn_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco/ga_faster_x101_32x4d_fpn_1x_coco_20200215-1ded9da3.pth";
        elif(model_name == "ga_faster_rcnn_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "ga_faster_rcnn_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco/ga_faster_x101_64x4d_fpn_1x_coco_20200215-0fa7bde7.pth";
        elif(model_name == "ga_retinanet_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "ga_retinanet_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth";
        elif(model_name == "ga_retinanet_x101_64x4d_fpn"):
            self.system_dict["local"]["model_name"] = "ga_retinanet_x101_64x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth";
        elif(model_name == "fovea_r50_fpn_4x4"):
            self.system_dict["local"]["model_name"] = "fovea_r50_fpn_4x4_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth";
        elif(model_name == "fovea_r101_fpn_4x4"):
            self.system_dict["local"]["model_name"] = "fovea_r101_fpn_4x4_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth";
        elif(model_name == "fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4"):
            self.system_dict["local"]["model_name"] = "fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth";
        elif(model_name == "fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4"):
            self.system_dict["local"]["model_name"] = "fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth";
        elif(model_name == "free_anchor_retinanet_r50_fpn"):
            self.system_dict["local"]["model_name"] = "free_anchor_retinanet_r50_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth";
        elif(model_name == "free_anchor_retinanet_r101_fpn"):
            self.system_dict["local"]["model_name"] = "free_anchor_retinanet_r101_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth";
        elif(model_name == "free_anchor_retinanet_x101_32x4d_fpn"):
            self.system_dict["local"]["model_name"] = "retinanet_free_anchor_x101_32x4d_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth";
        elif(model_name == "atss_r50_fpn"):
            self.system_dict["local"]["model_name"] = "atss_r50_fpn_coco";
            self.system_dict["params"]["load_from"] = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth";
        
    
     
    
    def Hyper_Params(self, lr=0.02, momentum=0.9, weight_decay=0.0001):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        
    def Training_Params(self, num_epochs=2, val_interval=1):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["val_interval"] = val_interval;
        
        self.update_config();
        
    def update_config(self):
        if(self.system_dict["local"]["model_name"] == "faster_rcnn_fpn50_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/faster_rcnn_fpn50_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "faster_rcnn_fpn101_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/faster_rcnn_fpn101_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "faster_rcnn_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/faster_rcnn_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "faster_rcnn_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/faster_rcnn_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "cascade_rcnn_fpn50_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/cascade_rcnn_fpn50_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "cascade_rcnn_fpn101_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/cascade_rcnn_fpn101_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "cascade_rcnn_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/cascade_rcnn_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "cascade_rcnn_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/cascade_rcnn_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_r50_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_r50_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_r101_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_r101_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_ghm_r50_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_ghm_r50_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_ghm_r101_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_ghm_r101_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_ghm_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_ghm_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_ghm_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_ghm_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "dh_faster_rcnn_fpn50_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/dh_faster_rcnn_fpn50_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "libra_faster_rcnn_fpn50_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/libra_faster_rcnn_fpn50_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "libra_faster_rcnn_fpn101_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/libra_faster_rcnn_fpn101_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "libra_faster_rcnn_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/libra_faster_rcnn_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "libra_retinanet_r50_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/libra_retinanet_r50_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "ga_faster_rcnn_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/ga_faster_rcnn_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "ga_faster_rcnn_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/ga_faster_rcnn_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "ga_retinanet_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/ga_retinanet_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "ga_retinanet_x101_64x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/ga_retinanet_x101_64x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "fovea_r50_fpn_4x4_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/fovea_r50_fpn_4x4_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "fovea_r101_fpn_4x4_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/fovea_r101_fpn_4x4_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "free_anchor_retinanet_r50_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/free_anchor_retinanet_r50_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "free_anchor_retinanet_r101_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/free_anchor_retinanet_r101_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "retinanet_free_anchor_x101_32x4d_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/retinanet_free_anchor_x101_32x4d_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        elif(self.system_dict["local"]["model_name"] == "atss_r50_fpn_coco"):
            f = open("Monk_Object_Detection/16_mmdet/lib/cfgs/atss_r50_fpn_coco.py");
            lines = f.read();
            f.close();
            
            lines = lines.replace("samples_per_gpu=",
                                  "samples_per_gpu=" + str(self.system_dict["params"]["batch_size"]));
            lines = lines.replace("workers_per_gpu=",
                                  "workers_per_gpu=" + str(self.system_dict["params"]["num_workers"]));
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["train_anno_file"]) + "',", 1);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 2);
            lines = lines.replace("ann_file=,",
                                  "ann_file='" + str(self.system_dict["params"]["val_anno_file"]) + "',", 3);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["train_img_folder"]) + "/',", 1);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 2);
            lines = lines.replace("img_prefix=,",
                                  "img_prefix='" + str(self.system_dict["params"]["val_img_folder"]) + "/',", 3);
            lines = lines.replace("evaluation = dict(interval=",
                                  "evaluation = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["val_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=,",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]) + ",");
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("load_from =",
                                  "load_from = '" + str(self.system_dict["params"]["load_from"]) + "'");
            lines = lines.replace("classes=,",
                                  "classes=" + str(self.system_dict["params"]["classes"]) + ",");
            lines = lines.replace("num_classes=80",
                                  "num_classes=" + str(len(self.system_dict["params"]["classes"])));

            if(self.system_dict["params"]["num_epochs"] >= 3):
                steps = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3];
            else:
                steps = [1];
            lines = lines.replace("step=",
                                  "step=" + str(steps));

            f = open("config_updated.py", 'w');
            f.write(lines);
            f.close();
        

             
        
    def Train(self):
        self.setup();
        
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(self.system_dict["local"]["cfg"].work_dir))
        # dump config
        self.system_dict["local"]["cfg"].dump(osp.join(self.system_dict["local"]["cfg"].work_dir, 
                                                       osp.basename(self.system_dict["params"]["config"])))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.system_dict["local"]["cfg"].work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.system_dict["local"]["cfg"].log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        
        # log some basic info
        logger.info(f'Distributed training: {self.system_dict["local"]["distributed"]}')
        logger.info(f'Config:\n{self.system_dict["local"]["cfg"].pretty_text}')

        # set random seeds
        if self.system_dict["params"]["seed"] is not None:
            logger.info(f'Set random seed to {self.system_dict["params"]["seed"]}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(self.system_dict["params"]["seed"], 
                            deterministic=self.system_dict["params"]["deterministic"])
        self.system_dict["local"]["cfg"].seed = self.system_dict["params"]["seed"]
        meta['seed'] = self.system_dict["params"]["seed"]
        
        model = build_detector(
            self.system_dict["local"]["cfg"].model, 
            train_cfg=self.system_dict["local"]["cfg"].train_cfg, 
            test_cfg=self.system_dict["local"]["cfg"].test_cfg)

        datasets = [build_dataset(self.system_dict["local"]["cfg"].data.train)]
        
        
        if len(self.system_dict["local"]["cfg"].workflow) == 2:
            val_dataset = copy.deepcopy(self.system_dict["local"]["cfg"].data.val)
            val_dataset.pipeline = self.system_dict["local"]["cfg"].data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        
        if self.system_dict["local"]["cfg"].checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            self.system_dict["local"]["cfg"].checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                config=self.system_dict["local"]["cfg"].pretty_text,
                CLASSES=datasets[0].CLASSES)
        
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            self.system_dict["local"]["cfg"],
            distributed=self.system_dict["local"]["distributed"],
            validate=(not self.system_dict["params"]["no_validate"]),
            timestamp=timestamp,
            meta=meta)
        
        
        
        
    def setup(self):
        self.system_dict["params"]["config"] = "config_updated.py";
        self.system_dict["params"]["work_dir"] = None;
        self.system_dict["params"]["resume_from"] = None;
        self.system_dict["params"]["no_validate"] = not self.system_dict["params"]["val_dataset"];
        self.system_dict["params"]["seed"] = None;
        self.system_dict["params"]["deterministic"] = False;
        self.system_dict["params"]["options"] = None;
        self.system_dict["params"]["cfg_options"] = None;
        self.system_dict["params"]["launcher"] = 'none';
        self.system_dict["params"]["local_rank"] = 0;
        
        
        self.system_dict["local"]["cfg"] = Config.fromfile(self.system_dict["params"]["config"])
        if self.system_dict["params"]["cfg_options"] is not None:
            self.system_dict["local"]["cfg"].merge_from_dict(self.system_dict["params"]["cfg_options"])
        # set cudnn_benchmark
        if self.system_dict["local"]["cfg"].get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined in this priority: CLI > segment in file > filename
        if self.system_dict["params"]["work_dir"] is not None:
            self.system_dict["local"]["cfg"].work_dir = self.system_dict["params"]["work_dir"]
        elif self.system_dict["local"]["cfg"].get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            self.system_dict["local"]["cfg"].work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(self.system_dict["params"]["config"]))[0])
        if self.system_dict["params"]["resume_from"] is not None:
            self.system_dict["local"]["cfg"].resume_from = self.system_dict["params"]["resume_from"]
        if self.system_dict["params"]["gpu_ids"] is not None:
            self.system_dict["local"]["cfg"].gpu_ids = self.system_dict["params"]["gpu_ids"]
        else:
            self.system_dict["local"]["cfg"].gpu_ids = range(1) if self.system_dict["params"]["gpus"] is None else range(self.system_dict["params"]["gpus"])

        # init distributed env first, since logger depends on the dist info.
        if self.system_dict["params"]["launcher"] == 'none':
            self.system_dict["local"]["distributed"] = False
        else:
            self.system_dict["local"]["distributed"] = True
            init_dist(self.system_dict["params"]["launcher"], **self.system_dict["local"]["cfg"].dist_params)
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        