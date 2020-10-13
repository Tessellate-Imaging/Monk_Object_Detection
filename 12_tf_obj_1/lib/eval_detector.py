
import os
import sys
import json
import numpy as np
from create_tfrecord import *

class Evaluator():
    def __init__(self, verbose=1):
        self.system_dict = {};

    
        
    def update_config(self, checkpoint_dir, num_classes, lr, decay, label_map, output_path, batch_size, num_steps):
        f = open(checkpoint_dir + "/pipeline.config");
        data = f.read();
        f.close();

        data = data.replace("input_path: \"" + output_path + "/train.record\"", 
                            "input_path: \"" + output_path + "/external_val.record\"");

        data = data.replace("input_path: \"" + output_path + "/val.record\"", 
                            "input_path: \"" + output_path + "/external_val.record\"");

        f = open(checkpoint_dir + "/pipeline_updated.config", 'w');
        f.write(data);
        f.close();
        
        
        
            
         
    
    
    def list_models(self):
        self.system_dict["model_list"] = ["ssd_mobilenet_v1", "ssd_mobilenet_v2", 
                                          "ssd_mobilenet_v1_ppn", "ssd_mobilenet_v1_fpn",
                                          "ssd_resnet50_v1_fpn", "ssd_mobilenet_v1_0.75_depth",
                                          "ssd_mobilenet_v1_quantized", "ssd_mobilenet_v1_0.75_depth_quantized",
                                          "ssd_mobilenet_v2_quantized", "ssdlite_mobilenet_v2",
                                          "ssd_inception_v2", "faster_rcnn_inception_v2",
                                          "faster_rcnn_resnet50", "faster_rcnn_resnet50_lowproposals",
                                          "rfcn_resnet101", "faster_rcnn_resnet101",
                                          "faster_rcnn_resnet101_lowproposals", "faster_rcnn_inception_resnet_v2_atrous",
                                          "faster_rcnn_inception_resnet_v2_atrous_lowproposals", "faster_rcnn_nas",
                                          "faster_rcnn_nas_lowproposals", "ssd_mobilenet_v2_mnasfpn",
                                          "ssd_mobilenet_v3_large", "ssd_mobilenet_v3_small"
                                         ];
        for i in range(len(self.system_dict["model_list"])):
            print("{}. Model Name: {}".format(i+1, self.system_dict["model_list"][i]));
        
              
        
    def set_val_dataset(self, img_dir, label_dir, class_list_file, only_eval=True):
        self.system_dict["val_img_dir"] = img_dir;
        self.system_dict["val_anno_dir"] = label_dir;
        self.system_dict["class_list_file"] = class_list_file;
        self.system_dict["only_eval"] = only_eval;

        
        
    def create_tfrecord(self, data_output_dir="data_tfrecord_for validation"):
        self.system_dict["output_path"] = data_output_dir;
        
        with open('system_dict_val.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
        
        create_val();
        
        with open('system_dict_val.json') as json_file:
            self.system_dict = json.load(json_file)
        
    
    def set_model_params(self, checkpoint_dir=None):        
        self.system_dict["checkpoint_dir"] = checkpoint_dir;
        with open('system_dict_val.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)

        self.set_hyper_params();

    
    def set_hyper_params(self,
                         sample_1_of_n_eval_examples=1,
                         sample_1_of_n_eval_on_train_examples=5):
        
        self.system_dict["num_train_steps"] = 10000;
        self.system_dict["lr"] = 0.004;
        self.system_dict["lr_decay_rate"] = 0.945;
        #self.system_dict["model_dir"] = output_dir_val;
        self.system_dict["sample_1_of_n_eval_examples"] = sample_1_of_n_eval_examples;
        self.system_dict["sample_1_of_n_eval_on_train_examples"] = sample_1_of_n_eval_on_train_examples;
        self.system_dict["run_once"] = 1;
        self.system_dict["max_eval_retries"] = 0;
            
        
        
        self.update_config(self.system_dict["checkpoint_dir"], 
                              self.system_dict["num_classes"], 
                              self.system_dict["lr"], 
                              self.system_dict["lr_decay_rate"],
                              self.system_dict["label_map"],
                              self.system_dict["output_path"],
                              1,
                              self.system_dict["num_train_steps"]);
    
        self.system_dict["pipeline_config_path"] = self.system_dict["checkpoint_dir"] + "/pipeline_updated.config";
        
        with open('system_dict_val.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
    


    
