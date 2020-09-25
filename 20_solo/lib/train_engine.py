import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger


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
        
    def Dataset_Params(self, batch_size=2, num_workers=2):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
    
        
    def List_Models(self):
        self.system_dict["params"]["model_list"] = ["solo_resnet50", "solo_resnet101"]
        
        for i in range(len(self.system_dict["params"]["model_list"])):
            print("{}. Model - {}".format(i+1, self.system_dict["params"]["model_list"][i]));  
        
    
    def Model_Params(self, model_name="solo_resnet50", gpu_devices=[0], launcher="none"):
        self.system_dict["local"]["model_name"] = model_name;
        self.system_dict["params"]["gpus"] = len(gpu_devices);
        self.system_dict["params"]["gpu_ids"] = gpu_devices;
        self.system_dict["params"]["launcher"] = launcher;
        
        if(model_name == "solo_resnet50"):
            if(not os.path.isfile("solo_resnet50_pretrained.pth")):
                print("Downloading Model (Takes around 20 mins) ...");
                os.system("wget https://cloudstor.aarnet.edu.au/plus/s/x4Fb4XQ0OmkBvaQ/download -O solo_resnet50_pretrained.pth")
                print("Done...");
        elif(model_name == "solo_resnet101"):
            if(not os.path.isfile("solo_resnet101_pretrained.pth")):
                print("Downloading Model (Takes around 30 mins) ...");
                os.system("wget https://cloudstor.aarnet.edu.au/plus/s/WxOFQzHhhKQGxDG/download -O solo_resnet50_pretrained.pth")
                print("Done...");
        elif(model_name == "decoupled_solo_resnet50"):
            if(not os.path.isfile("solo_resnet50_pretrained.pth")):
                print("Downloading Model (Takes around 30 mins) ...");
                os.system("wget https://cloudstor.aarnet.edu.au/plus/s/dXz11J672ax0Z1Q/download -O solo_resnet50_pretrained.pth")
                print("Done...");
        
        
    def Hyper_Params(self, lr=0.001, momentum=0.9, weight_decay=0.0001, 
                     autoscale_lr=False, seed=None, deterministic=False):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        self.system_dict["params"]["autoscale_lr"] = autoscale_lr;
        self.system_dict["params"]["seed"] = seed;
        self.system_dict["params"]["deterministic"] = deterministic;
        
    def Training_Params(self, num_epochs=2, save_interval=1):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["save_interval"] = save_interval;
        
        self.update_config();
        
    def update_config(self):
        if(self.system_dict["local"]["model_name"] == "solo_resnet50"):
            f = open("Monk_Object_Detection/20_solo/lib/configs_base/solo_resnet50.py");
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
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["save_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("num_classes=81",
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
        elif(self.system_dict["local"]["model_name"] == "solo_resnet101"):
            f = open("Monk_Object_Detection/20_solo/lib/configs_base/solo_resnet101.py");
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
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["save_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("num_classes=81",
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
        elif(self.system_dict["local"]["model_name"] == "decoupled_solo_resnet50"):
            f = open("Monk_Object_Detection/20_solo/lib/configs_base/decoupled_solo_resnet50.py");
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
            lines = lines.replace("checkpoint_config = dict(interval=",
                                  "checkpoint_config = dict(interval=" + str(self.system_dict["params"]["save_interval"]));
            lines = lines.replace("lr=",
                                  "lr=" + str(self.system_dict["params"]["lr"]));
            lines = lines.replace("momentum=",
                                  "momentum=" + str(self.system_dict["params"]["momentum"]));
            lines = lines.replace("weight_decay=",
                                  "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));
            lines = lines.replace("total_epochs =",
                                  "total_epochs =" + str(self.system_dict["params"]["num_epochs"]));
            lines = lines.replace("num_classes=81",
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
        
        train_detector(
            self.system_dict["local"]["model"],
            self.system_dict["local"]["datasets"],
            self.system_dict["local"]["cfg"],
            distributed=self.system_dict["local"]["distributed"],
            validate=False,
            timestamp=self.system_dict["local"]["timestamp"])
        
        
    def setup(self):
        cfg = Config.fromfile("config_updated.py")
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        # update configs according to CLI args
        cfg.gpus = self.system_dict["params"]["gpus"]

        
        if self.system_dict["params"]["autoscale_lr"]:
            # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

        # init distributed env first, since logger depends on the dist info.
        if self.system_dict["params"]["launcher"] == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(self.system_dict["params"]["launcher"], **cfg.dist_params)
            
        
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # log some basic info
        logger.info('Distributed training: {}'.format(distributed))
        logger.info('MMDetection Version: {}'.format(__version__))
        logger.info('Config:\n{}'.format(cfg.text))

        # set random seeds
        if self.system_dict["params"]["seed"] is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(
                self.system_dict["params"]["seed"], self.system_dict["params"]["deterministic"]))
            set_random_seed(self.system_dict["params"]["seed"], deterministic=self.system_dict["params"]["deterministic"])
        
        
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        datasets = [build_dataset(cfg.data.train)]
        
        if len(cfg.workflow) == 2:
            datasets.append(build_dataset(cfg.data.val))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=self.system_dict["params"]["classes"])
        # add an attribute for visualization convenience
        model.CLASSES = self.system_dict["params"]["classes"]
        
        
        
        
        
        self.system_dict["local"]["cfg"] = cfg;
        self.system_dict["local"]["distributed"] = distributed;
        self.system_dict["local"]["logger"] = logger;
        self.system_dict["local"]["timestamp"] = timestamp;
        self.system_dict["local"]["datasets"] = datasets;
        self.system_dict["local"]["model"] = model;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        