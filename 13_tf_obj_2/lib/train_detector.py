import os
import sys
import json
import numpy as np
from create_tfrecord import *

class Detector():
    def __init__(self, verbose=1):
        self.system_dict = {};
        
    def download_model(self, model_name):
        if(model_name == "ssd_mobilenet_v2"):
            model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_mobilenet_v1_fpn"):
            model_name = "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_mobilenet_v2_fpnlite"):
            model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_mobilenet_v2_fpnlite_v2"):
            model_name = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_resnet50_v1_fpn"):
            model_name = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_resnet50_v1_fpn_v2"):
            model_name = "ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_resnet101_v1_fpn"):
            model_name = "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_resnet101_v1_fpn_v2"):
            model_name = "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_resnet152_v1_fpn"):
            model_name = "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz")
            return model_name;
        elif(model_name == "ssd_resnet152_v1_fpn_v2"):
            model_name = "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz")
                os.system("tar -xvzf ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz")
            return model_name;
        
        
        
        
        
     
    
    def update_config(self, model_name, num_classes, lr, decay, label_map, output_path, batch_size, num_steps):
        if(model_name == "ssd_mobilenet_v2_320x320_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 512", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.800000011920929", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.13333000242710114", 
                                "warmup_learning_rate: " + str(lr/5));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 128", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.07999999821186066", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.026666000485420227", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 128", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.07999999821186066", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.026666000485420227", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
        elif(model_name == "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8"):
            f = open(model_name + "/pipeline.config");
            data = f.read();
            f.close();

            data = data.replace("num_classes: 90", 
                                "num_classes: " + str(num_classes));
            
            data = data.replace("batch_size: 64", 
                                "batch_size: " + str(batch_size));
            
            
            data = data.replace("learning_rate_base: 0.03999999910593033", 
                                "learning_rate_base: " + str(lr));
            
            data = data.replace("warmup_learning_rate: 0.013333000242710114", 
                                "warmup_learning_rate: " + str(lr/3));
            
            data = data.replace("fine_tune_checkpoint_type: \"classification\"", 
                                "fine_tune_checkpoint_type: \"detection\"");
            
            
            
            data = data.replace("fine_tune_checkpoint: \"PATH_TO_BE_CONFIGURED\"", 
                                "fine_tune_checkpoint: \"" + model_name + "/checkpoint/ckpt-0\"");

            data = data.replace("label_map_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "label_map_path: \"" + label_map + "\"");

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/train.record\"", 1);

            data = data.replace("input_path: \"PATH_TO_BE_CONFIGURED\"", 
                                "input_path: \"" + output_path + "/val.record\"", 2);

            
            f = open(model_name + "/pipeline_updated.config", 'w');
            f.write(data);
            f.close();
            
        
        
          
            
    
    def list_models(self):
        self.system_dict["model_list"] = ["ssd_mobilenet_v2", "ssd_mobilenet_v1_fpn",
                                          "ssd_mobilenet_v2_fpnlite", "ssd_mobilenet_v2_fpnlite_v2",
                                          "ssd_resnet50_v1_fpn", "ssd_resnet50_v1_fpn_v2",
                                          "ssd_resnet101_v1_fpn", "ssd_resnet101_v1_fpn_v2",
                                          "ssd_resnet152_v1_fpn", "ssd_resnet152_v1_fpn_v2"
                                         ];
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
        print("Downloading Model");
        self.system_dict["model_name"] = self.download_model(self.system_dict["model_name"]);
        print("Model Download");
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
                         max_eval_retries=0,
                         num_workers=4,
                         checkpoint_after_every=500):
        
        self.system_dict["num_train_steps"] = num_train_steps;
        self.system_dict["lr"] = lr;
        self.system_dict["lr_decay_rate"] = lr_decay_rate;
        self.system_dict["model_dir"] = output_dir;
        self.system_dict["sample_1_of_n_eval_examples"] = sample_1_of_n_eval_examples;
        self.system_dict["sample_1_of_n_eval_on_train_examples"] = sample_1_of_n_eval_on_train_examples;
        self.system_dict["checkpoint_dir"] = checkpoint_dir;
        self.system_dict["run_once"] = run_once;
        self.system_dict["max_eval_retries"] = max_eval_retries;
        self.system_dict["eval_timeout"] = 3600
        self.system_dict["use_tpu"] = False;
        self.system_dict["num_workers"] = num_workers
        self.system_dict["tpu_name"] = None
        self.system_dict["checkpoint_every_n"] = checkpoint_after_every;
        self.system_dict["record_summaries"] = True;
        
            
        
        
        self.update_config(self.system_dict["model_name"], 
                              self.system_dict["num_classes"], 
                              self.system_dict["lr"], 
                              self.system_dict["lr_decay_rate"],
                              self.system_dict["label_map"],
                              self.system_dict["output_path"],
                              self.system_dict["batch_size"],
                              self.system_dict["num_train_steps"]);
    
        self.system_dict["pipeline_config_path"] = self.system_dict["model_name"] + "/pipeline_updated.config";
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(self.system_dict, json_file)
            
     
    def export_params(self, output_directory="export_dir"):
        self.system_dict["input_type"] = "image_tensor";
        
        if(self.system_dict["model_name"] == "ssd_mobilenet_v2_320x320_coco17_tpu-8" or
           self.system_dict["model_name"] == "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
           ):
            self.system_dict["input_shape"] = "-1, 320, 320, 3";
            self.system_dict["input_shape_flops"] = "1, 320, 320, 3";
        elif(self.system_dict["model_name"] == "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8" or
             self.system_dict["model_name"] == "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8" or
             self.system_dict["model_name"] == "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8" or
             self.system_dict["model_name"] == "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"
            ):
            self.system_dict["input_shape"] = "-1, 640, 640, 3";
            self.system_dict["input_shape_flops"] = "1, 640, 640, 3";
        elif(self.system_dict["model_name"] == "ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8" or
             self.system_dict["model_name"] == "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8" or
             self.system_dict["model_name"] == "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8" or
             self.system_dict["model_name"] == "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8"
            ):
            self.system_dict["input_shape"] = "-1, 1024, 1024, 3";
            self.system_dict["input_shape_flops"] = "1, 1024, 1024, 3";
        
        
        
        self.system_dict["trained_checkpoint_dir"] = self.system_dict["model_dir"];
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
            
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            