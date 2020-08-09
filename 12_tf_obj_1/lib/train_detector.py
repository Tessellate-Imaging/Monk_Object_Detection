import os
import sys
import json
import numpy as np
from create_tfrecord import *

class Detector():
    def __init__(self, verbose=1):
        self.system_dict = {};
        
    def download_model(self, model_name):
        if(model_name == "ssd_mobilenet_v1"):
            model_name = "ssd_mobilenet_v1_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz")
                os.system("tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz")
            return model_name;
        elif(model_name == "ssd_mobilenet_v2"):
            model_name = "ssd_mobilenet_v2_coco_2018_03_29";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
                os.system("tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
            return model_name;
        elif(model_name == "ssd_mobilenet_v1_ppn"):
            model_name = "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v1_fpn"):
            model_name = "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz");
            return model_name;
        elif(model_name == "ssd_resnet50_v1_fpn"):
            model_name = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz");
                os.system("tar -xvzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v1_0.75_depth"):
            model_name = "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz ");
                os.system("tar -xvzf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v1_quantized"):
            model_name = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v1_0.75_depth_quantized"):
            model_name = "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz");
            return model_name;
            
        
        
        
    def update_config(self, model_name, num_classes, lr, decay, label_map, output_path, batch_size):
        if(model_name == "ssd_mobilenet_v1_coco_2018_01_28"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("initial_learning_rate: 0.00400000018999", 
                                "initial_learning_rate: " + str(lr));

            data = data.replace("decay_factor: 0.949999988079", 
                                "decay_factor: " + str(decay));

            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 24", 
                                "batch_size: " + str(batch_size));
            
            data = data.replace("batch_norm_trainable: true", 
                                "#batch_norm_trainable: true");

            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        
        elif(model_name == "ssd_mobilenet_v2_coco_2018_03_29"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("initial_learning_rate: 0.00400000018999", 
                                "initial_learning_rate: " + str(lr));

            data = data.replace("decay_factor: 0.949999988079", 
                                "decay_factor: " + str(decay));

            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 24", 
                                "batch_size: " + str(batch_size));
            
            data = data.replace("batch_norm_trainable: true", 
                                "#batch_norm_trainable: true");

            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close()
            
        elif(model_name == "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("learning_rate_base: 0.699999988079", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.13330000639", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 512", 
                                "batch_size: " + str(batch_size));
            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
            
        elif(model_name == "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("learning_rate_base: 0.0799999982119", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.0266660004854", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 128", 
                                "batch_size: " + str(batch_size));
            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
            
        elif(model_name == "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("learning_rate_base: 0.0399999991059", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.0133330002427", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
            
        elif(model_name == "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("learning_rate_base: 0.899999976158", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.300000011921", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 2048", 
                                "batch_size: " + str(batch_size));
            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
            
        elif(model_name == "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("learning_rate_base: 0.20000000298", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.0599999986589", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 128", 
                                "batch_size: " + str(batch_size));
            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
            
        elif(model_name == "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            data = data.replace("learning_rate_base: 0.20000000298", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.0599999986589", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED/model.ckpt\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/model.ckpt\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100\"", 
                                "input_path: \"" + output_path + "/train.record\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010\"", 
                                "input_path: \"" + output_path + "/val.record\"");

            data = data.replace("batch_size: 128", 
                                "batch_size: " + str(batch_size));
            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        
        
        
        

    
    def list_models(self):
        self.system_dict["model_list"] = ["ssd_mobilenet_v1", "ssd_mobilenet_v2", 
                                          "ssd_mobilenet_v1_ppn", "ssd_mobilenet_v1_fpn",
                                          "ssd_resnet50_v1_fpn", "ssd_mobilenet_v1_0.75_depth",
                                          "ssd_mobilenet_v1_quantized", "ssd_mobilenet_v1_0.75_depth_quantized"];
        for i in range(len(self.system_dict["model_list"])):
            print("{}. Model Name: {}".format(i+1, self.system_dict["model_list"][i]));
        
        
        
    def set_train_dataset(self, img_dir, label_dir, class_list_file, batch_size=20, trainval_split = 0.8):
        self.system_dict["train_img_dir"] = img_dir;
        self.system_dict["train_anno_dir"] = label_dir;
        self.system_dict["val_img_dir"] = False;
        self.system_dict["val_anno_dir"] = False;
        self.system_dict["batch_size"] = batch_size;
        self.system_dict["trainval_split"] = trainval_split;
        self.system_dict["class_list_file"] = class_list_file;
        
        
    def set_val_dataset(self, img_dir, label_dir):
        self.system_dict["val_img_dir"] = img_dir;
        self.system_dict["val_anno_dir"] = label_dir;
        
        
    def create_tfrecord(self, data_output_dir="data_tfrecord"):
        self.system_dict["output_path"] = data_output_dir;
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
        
        create();
        
        with open('system_dict.json') as json_file:
            self.system_dict = json.load(json_file)
        
    
    def set_model_params(self, model_name="ssd_mobilenet_v1"):
        self.system_dict["model_name"] = model_name;
        self.system_dict["model_name"] = self.download_model(self.system_dict["model_name"]);
        print("Model name set as {}".format(self.system_dict["model_name"]));
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
            
    
    def set_hyper_params(self, 
                         num_train_steps=10000,
                         lr=0.004,
                         lr_decay_rate=0.945,
                         output_dir="output_dir/",
                         sample_1_of_n_eval_examples=1,
                         sample_1_of_n_eval_on_train_examples=5,
                         checkpoint_dir=False,
                         run_once=False,
                         max_eval_retries=0):
        
        self.system_dict["num_train_steps"] = num_train_steps;
        self.system_dict["lr"] = lr;
        self.system_dict["lr_decay_rate"] = lr_decay_rate;
        self.system_dict["model_dir"] = output_dir;
        self.system_dict["sample_1_of_n_eval_examples"] = sample_1_of_n_eval_examples;
        self.system_dict["sample_1_of_n_eval_on_train_examples"] = sample_1_of_n_eval_on_train_examples;
        self.system_dict["checkpoint_dir"] = checkpoint_dir;
        self.system_dict["run_once"] = run_once;
        self.system_dict["max_eval_retries"] = max_eval_retries;
            
        
        
        self.update_config(self.system_dict["model_name"], 
                              self.system_dict["num_classes"], 
                              self.system_dict["lr"], 
                              self.system_dict["lr_decay_rate"],
                              self.system_dict["label_map"],
                              self.system_dict["output_path"],
                              self.system_dict["batch_size"]);
    
        self.system_dict["pipeline_config_path"] = self.system_dict["model_name"] + "/pipeline_updated.config";
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
            
     
    def export_params(self, output_directory="export_dir"):
        self.system_dict["input_type"] = "image_tensor";
        
        if(self.system_dict["model_name"] == "ssd_mobilenet_v1_coco_2018_01_28" or
           self.system_dict["model_name"] == "ssd_mobilenet_v2_coco_2018_03_29" or
           self.system_dict["model_name"] == "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03" or
           self.system_dict["model_name"] == "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03" or
           self.system_dict["model_name"] == "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18" or
           self.system_dict["model_name"] == "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18"):
            self.system_dict["input_shape"] = "-1, 300, 300, 3";
        elif(self.system_dict["model_name"] == "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03" or
            self.system_dict["model_name"] == "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"):
            self.system_dict["input_shape"] = "-1, 640, 640, 3";
            
        
        
        self.system_dict["trained_checkpoint_prefix"] = self.system_dict["model_dir"] + "/model.ckpt-" + str(self.system_dict["num_train_steps"]);
        self.system_dict["trained_checkpoint_prefix"] = self.system_dict["trained_checkpoint_prefix"].replace("//", "/");
        self.system_dict["output_directory"] = output_directory;
        self.system_dict["config_override"] = "";
        self.system_dict["write_inference_graph"] = False;
        self.system_dict["additional_output_tensor_names"] = None;
        self.system_dict["use_side_inputs"] = False;
        self.system_dict["side_input_shapes"] = None;
        self.system_dict["side_input_types"] = None;
        self.system_dict["side_input_names"] = None;
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
    
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
        
    