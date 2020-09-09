import os
import os.path as osp

from mmcv import Config
from mmcv.runner import set_random_seed

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv


class Detector_Videos():
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
        
    def Train_Video_Dataset(self, img_folder, anno_file, classes_file):
        self.system_dict["params"]["train_img_folder"] = img_folder;
        self.system_dict["params"]["train_anno_file"] = anno_file;
        self.system_dict["params"]["classes_file"] = classes_file;
        
        f = open(self.system_dict["params"]["classes_file"]);
        lines = f.readlines();
        f.close();
        self.system_dict["params"]["classes"] = [];
        for i in range(len(lines)):
            if(lines[i] != ""):
                self.system_dict["params"]["classes"].append(lines[i][:len(lines[i])-1])
        self.system_dict["params"]["num_classes"] = len(self.system_dict["params"]["classes"]);
        
        #Dummy
        self.system_dict["params"]["val_img_folder"] = img_folder;
        self.system_dict["params"]["val_anno_file"] = anno_file;
        
        
    def Val_Video_Dataset(self, img_folder, anno_file):
        self.system_dict["params"]["val_img_folder"] = img_folder;
        self.system_dict["params"]["val_anno_file"] = anno_file;
        self.system_dict["params"]["val_dataset"] = True;
        
    def Dataset_Params(self, batch_size=2, num_workers=2):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
        
    def List_Models(self):
        self.system_dict["params"]["model_list"] = ["tsn_r50", "tsm_r50"];
        
        for i in range(len(self.system_dict["params"]["model_list"])):
            model_name = self.system_dict["params"]["model_list"][i];
            print("{}. Model - {}".format(i+1, model_name));
                
    def Model_Params(self, model_name="tsn_r50", gpu_devices=[0]):
        self.system_dict["params"]["gpus"] = len(gpu_devices);
        self.system_dict["params"]["gpu_ids"] = gpu_devices;
        if(len(gpu_devices) > 1):
            self.system_dict["params"]["distributed"] = True;
        else:
            self.system_dict["params"]["distributed"] = False;
        
        if(model_name == "tsn_r50"):
            self.system_dict["params"]["model_name"] = "tsn_r50_video_1x1x8_100e_kinetics400_rgb";
            self.system_dict["params"]["config_file"] = "Monk_Object_Detection/18_mmaction/lib/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py";
            self.system_dict["params"]["inference_config_file"] = "Monk_Object_Detection/18_mmaction/lib/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py";
            self.system_dict["params"]["load_from"] = "https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth";
                                                    
        elif(model_name == "tsm_r50"):
            self.system_dict["params"]["model_name"] = "tsm_r50_video_1x1x8_50e_kinetics400_rgb";
            self.system_dict["params"]["config_file"] = "Monk_Object_Detection/18_mmaction/lib/configs/recognition/tsm/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py";
            self.system_dict["params"]["inference_config_file"] = "Monk_Object_Detection/18_mmaction/lib/configs/recognition/tsm/tsm_r50_video_inference_1x1x8_100e_kinetics400_rgb.py";
            self.system_dict["params"]["load_from"] = "https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth";
            
    def Hyper_Params(self, lr=0.02, momentum=0.9, weight_decay=0.0001):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        
    def Training_Params(self, num_epochs=2, val_interval=1):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["val_interval"] = val_interval;
        
        self.system_dict["local"]["cfg"] = self.update_config();
        
    def update_config(self):
        cfg = Config.fromfile(self.system_dict["params"]["config_file"]);
        
        # Modify dataset type and path
        cfg.dataset_type = 'VideoDataset'
        cfg.data_root = self.system_dict["params"]["train_img_folder"]
        cfg.data_root_val = self.system_dict["params"]["val_img_folder"]
        cfg.ann_file_train = self.system_dict["params"]["train_anno_file"]
        cfg.ann_file_val = self.system_dict["params"]["val_anno_file"]
        cfg.ann_file_test = self.system_dict["params"]["val_anno_file"]

        cfg.data.test.type = 'VideoDataset'
        cfg.data.test.ann_file = self.system_dict["params"]["val_anno_file"]
        cfg.data.test.data_prefix = self.system_dict["params"]["val_img_folder"]

        cfg.data.train.type = 'VideoDataset'
        cfg.data.train.ann_file = self.system_dict["params"]["train_anno_file"]
        cfg.data.train.data_prefix = self.system_dict["params"]["train_img_folder"]

        cfg.data.val.type = 'VideoDataset'
        cfg.data.val.ann_file = self.system_dict["params"]["val_anno_file"]
        cfg.data.val.data_prefix = self.system_dict["params"]["val_img_folder"]

        # Modify num classes of the model in cls_head
        cfg.model.cls_head.num_classes = self.system_dict["params"]["num_classes"]
        # We can use the pre-trained TSN model
        cfg.load_from = self.system_dict["params"]["load_from"] 

        # Set up working dir to save files and logs.
        cfg.work_dir = './work_dirs'

        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        cfg.data.videos_per_gpu = self.system_dict["params"]["batch_size"]
        cfg.data.workers_per_gpu = self.system_dict["params"]["num_workers"]
        cfg.optimizer.lr = self.system_dict["params"]["lr"];
        cfg.optimizer.momentum = self.system_dict["params"]["momentum"];
        cfg.optimizer.weight_decay = self.system_dict["params"]["weight_decay"];
        cfg.total_epochs = self.system_dict["params"]["num_epochs"]
        if(self.system_dict["params"]["num_epochs"] > 2):
            cfg.lr_config.step = [self.system_dict["params"]["num_epochs"]//3,
                                  2*self.system_dict["params"]["num_epochs"]//3];
        else:
            cfg.lr_config.step = [1];
        

        # We can set the checkpoint saving interval to reduce the storage cost
        cfg.checkpoint_config.interval = self.system_dict["params"]["val_interval"]
        cfg.evaluation.interval = self.system_dict["params"]["val_interval"]
        # We can set the log print interval to reduce the the times of printing log
        cfg.log_config.interval = self.system_dict["params"]["val_interval"]

        # Set seed thus the results are more reproducible
        cfg.seed = 0
        set_random_seed(0, deterministic=False)
        cfg.gpu_ids = self.system_dict["params"]["gpu_ids"]
        
        return cfg;
    
    def Train(self):
        self.setup();
        print("Starting to train ...");
        train_model(self.system_dict["local"]["model"], 
                    self.system_dict["local"]["datasets"], 
                    self.system_dict["local"]["cfg"], 
                    distributed=self.system_dict["params"]["distributed"], 
                    validate=self.system_dict["params"]["val_dataset"])
        print("Done");
        print("Creating inference config file");
        cfg_infer = Config.fromfile(self.system_dict["params"]["inference_config_file"]);
        cfg_infer.model.cls_head.num_classes = self.system_dict["params"]["num_classes"]
        cfg_infer.dump(self.system_dict["local"]["cfg"].work_dir + "/config.py");
        print("Done");
        
        
    def setup(self):
        print("loading dataset ...");
        self.system_dict["local"]["datasets"] = [build_dataset(self.system_dict["local"]["cfg"].data.train)]

        print("loading_model ...");
        self.system_dict["local"]["model"] = build_model(self.system_dict["local"]["cfg"].model, 
                            train_cfg=self.system_dict["local"]["cfg"].train_cfg, 
                            test_cfg=self.system_dict["local"]["cfg"].test_cfg)

        print("creating workspace directory ...");
        mmcv.mkdir_or_exist(osp.abspath(self.system_dict["local"]["cfg"].work_dir))
        
        print("Done");
        
   
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    