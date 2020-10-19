import xmltodict
import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import time
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt
from calculate_map import *


from tensorflow.python.tools import freeze_graph

class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["coco_class"] = ["person",
                                            "bicycle",
                                            "car",
                                            "motorcycle",
                                            "airplane",
                                            "bus",
                                            "train",
                                            "truck",
                                            "boat",
                                            "traffic light",
                                            "fire hydrant",
                                            "street sign",
                                            "stop sign",
                                            "parking meter",
                                            "bench",
                                            "bird",
                                            "cat",
                                            "dog",
                                            "horse",
                                            "sheep",
                                            "cow",
                                            "elephant",
                                            "bear",
                                            "zebra",
                                            "giraffe",
                                            "hat",
                                            "backpack",
                                            "umbrella",
                                            "shoe",
                                            "eye glasses",
                                            "handbag",
                                            "tie",
                                            "suitcase",
                                            "frisbee",
                                            "skis",
                                            "snowboard",
                                            "sports ball",
                                            "kite",
                                            "baseball bat",
                                            "baseball glove",
                                            "skateboard",
                                            "surfboard",
                                            "tennis racket",
                                            "bottle",
                                            "plate",
                                            "wine glass",
                                            "cup",
                                            "fork",
                                            "knife",
                                            "spoon",
                                            "bowl",
                                            "banana",
                                            "apple",
                                            "sandwich",
                                            "orange",
                                            "broccoli",
                                            "carrot",
                                            "hot dog",
                                            "pizza",
                                            "donut",
                                            "cake",
                                            "chair",
                                            "couch",
                                            "potted plant",
                                            "bed",
                                            "mirror",
                                            "dining table",
                                            "window",
                                            "desk",
                                            "toilet",
                                            "door",
                                            "tv",
                                            "laptop",
                                            "mouse",
                                            "remote",
                                            "keyboard",
                                            "cell phone",
                                            "microwave",
                                            "oven",
                                            "toaster",
                                            "sink",
                                            "refrigerator",
                                            "blender",
                                            "book",
                                            "clock",
                                            "vase",
                                            "scissors",
                                            "teddy bear",
                                            "hair drier",
                                            "toothbrush",
                                            "hair brush"]

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
        elif(model_name == "ssd_mobilenet_v2_quantized"):
            model_name = "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz");
            return model_name;
        elif(model_name == "ssdlite_mobilenet_v2"):
            model_name = "ssdlite_mobilenet_v2_coco_2018_05_09";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz ");
                os.system("tar -xvzf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz");
            return model_name;
        elif(model_name == "ssd_inception_v2"):
            model_name = "ssd_inception_v2_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz ");
                os.system("tar -xvzf ssd_inception_v2_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_inception_v2"):
            model_name = "faster_rcnn_inception_v2_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz ");
                os.system("tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_resnet50"):
            model_name = "faster_rcnn_resnet50_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz ");
                os.system("tar -xvzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_resnet50_lowproposals"):
            model_name = "faster_rcnn_resnet50_lowproposals_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "rfcn_resnet101"):
            model_name = "rfcn_resnet101_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf rfcn_resnet101_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_resnet101"):
            model_name = "faster_rcnn_resnet101_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_resnet101_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_resnet101_lowproposals"):
            model_name = "faster_rcnn_resnet101_lowproposals_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_inception_resnet_v2_atrous"):
            model_name = "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_inception_resnet_v2_atrous_lowproposals"):
            model_name = "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_nas"):
            model_name = "faster_rcnn_nas_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget  http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_nas_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "faster_rcnn_nas_lowproposals"):
            model_name = "faster_rcnn_nas_lowproposals_coco_2018_01_28";
            if(not os.path.isdir(model_name)):
                os.system("wget  http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz");
                os.system("tar -xvzf faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v2_mnasfpn"):
            model_name = "ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18";
            if(not os.path.isdir(model_name)):
                os.system("wget  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v3_large"):
            model_name = "ssd_mobilenet_v3_large_coco_2020_01_14";
            if(not os.path.isdir(model_name)):
                os.system("wget  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz");
            return model_name;
        elif(model_name == "ssd_mobilenet_v3_small"):
            model_name = "ssd_mobilenet_v3_small_coco_2020_01_14";
            if(not os.path.isdir(model_name)):
                os.system("wget  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz");
                os.system("tar -xvzf ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz");
            return model_name;

    def list_pretrained_models(self):
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


    def set_pretrained_model_params(self, model_name="ssd_mobilenet_v1"):
        self.system_dict["classes"] = None;

        print("Downloading Model");
        self.system_dict["model_name"] = model_name;
        self.system_dict["model_name"] = self.download_model(self.system_dict["model_name"]);
        print("Model Download");
        print("Model name set as {}".format(self.system_dict["model_name"]));

        path_to_frozen_graph = self.system_dict["model_name"] + "/frozen_inference_graph.pb";
        with tf.gfile.FastGFile(path_to_frozen_graph, 'rb') as f:
            self.system_dict["graph_def"] = tf.GraphDef()
            self.system_dict["graph_def"].ParseFromString(f.read())
            
        self.system_dict["sess"] = tf.Session()
        self.system_dict["sess"].graph.as_default()
        tf.import_graph_def(self.system_dict["graph_def"], name='')
        
    
    def set_model_params(self, path_to_frozen_graph, class_list_file):
        self.system_dict["path_to_frozen_graph"] = path_to_frozen_graph;
        
        with tf.gfile.FastGFile(path_to_frozen_graph, 'rb') as f:
            self.system_dict["graph_def"] = tf.GraphDef()
            self.system_dict["graph_def"].ParseFromString(f.read())
            
        self.system_dict["sess"] = tf.Session()
        self.system_dict["sess"].graph.as_default()
        tf.import_graph_def(self.system_dict["graph_def"], name='')
        
        f = open(class_list_file, 'r');
        self.system_dict["classes"] = [];
        lines = f.readlines();
        f.close();
        for i in range(len(lines)):
            if(lines[i] != ""):
                self.system_dict["classes"].append(lines[i][:len(lines[i])-1])
        
        
    def infer_on_image(self, img_path, output_img_path="output.png", img_size=300, thresh=0.5, bbox_thickness=3, text_size=2, text_thickness=4, verbose=True):
        start = time.time();
        img = cv.imread(img_path)
        rows = img.shape[0]
        cols = img.shape[1]
        if(img_size):
            inp = cv.resize(img, (img_size, img_size))
        else:
            inp = img;
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        end = time.time();
        if(verbose):
            print("Image loaded in {} sec".format(end-start));

        # Run the model
        start = time.time();
        out = self.system_dict["sess"].run([self.system_dict["sess"].graph.get_tensor_by_name('num_detections:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_scores:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_boxes:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        end = time.time();
        if(verbose):
            print("Predicted in {} sec".format(end-start));
              
              
        start = time.time();   
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        # Iterate through all detected detections

        scores = [];
        bboxes = [];
        labels = [];
        for i in range(num_detections):
            classId = int(out[3][0][i])

            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if(self.system_dict["classes"]):
                label = self.system_dict["classes"][int(out[3][0][i])-1]
            else:
                label = self.system_dict["coco_class"][int(out[3][0][i])-1]

            if score > thresh:
                # Creating a box around the detected number plate
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                cv.rectangle(img, (x, y), (right, bottom), (0, 0, 255), thickness=bbox_thickness)
                cv.putText(img, label, (x,y), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 225), text_thickness)
                cv.imwrite(output_img_path, img)
                scores.append(score);
                bboxes.append([x, y, right, bottom])
                labels.append(label);
                
        end = time.time();
        if(verbose):
            print("Inference printed on image in {} sec".format(end-start));
        
        return scores, bboxes, labels;
    
    
    def infer_on_folder(self, folder_path, output_folder_path="output_folder", img_size=300, thresh=0.5, bbox_thickness=3, text_size=2, text_thickness=4):
        img_list = os.listdir(folder_path);
        output = [];
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path);

        for i in tqdm(range(len(img_list))):
            input_image = folder_path + "/" + img_list[i];
            output_image = output_folder_path + "/" + img_list[i];
            scores, bboxes, labels = self.infer_on_image(input_image, output_img_path=output_image, 
                                                    img_size=300, thresh=0.5, bbox_thickness=3, text_size=2, text_thickness=4, verbose=False);
            tmp = {};
            tmp["input_image"] = input_image;
            tmp["output_image"] = output_image;
            tmp["scores"] = scores;
            tmp["bboxes"] = bboxes;
            tmp["labels"] = labels;

            output.append(tmp);

        print("Outputs stored at folder - {}".format(output_folder_path));
        return output;

    def evaluate_on_folder(self, image_folder, gt_folder, output_folder_path='result', thresh=0.1, iou_thresh=0.5,
                            bbox_thickness=3, text_size=2, text_thickness=4):
        img_list = sorted(os.listdir(image_folder));
        anno_list = sorted(os.listdir(gt_folder));

        output = self.infer_on_folder(image_folder, output_folder_path=output_folder_path, img_size=300, thresh=0.5, bbox_thickness=3, text_size=2, text_thickness=4);

        if(os.path.isdir("tmp")):
            os.system("rm -r tmp");

        os.mkdir("tmp");
        os.mkdir("tmp/gt");
        os.mkdir("tmp/result");

        for i in tqdm(range(len(output))):
            fname = output[i]["input_image"].split("/")[-1].split(".")[0];
            bboxes = output[i]["bboxes"];
            scores = output[i]["scores"];
            labels = output[i]["labels"];
            
            f = open("tmp/result/" + fname + ".txt", 'w');
            for j in range(len(bboxes)):
                label = labels[j].replace(" ", "_");
                x1, y1, x2, y2 = bboxes[j];
                wr = label + " " + str(scores[j]) + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n";
                f.write(wr);
            
            f.close();

        anno_list = os.listdir(gt_folder);

        for i in tqdm(range(len(anno_list))):
            annoFile = gt_folder + '/' + anno_list[i];
            f = open(annoFile, 'r');
            my_xml = f.read();
            anno = dict(dict(xmltodict.parse(my_xml))["annotation"])
            f.close();
            fname = anno["filename"];
            
            f = open("tmp/gt/" + anno_list[i].split(".")[0] + ".txt", 'w');
            
            if(type(anno["object"]) == list):
                for j in range(len(anno["object"])):
                    obj = dict(anno["object"][j]);
                    label = anno["object"][j]["name"];
                    bbox = dict(anno["object"][j]["bndbox"])
                    x1 = bbox["xmin"];
                    y1 = bbox["ymin"];
                    x2 = bbox["xmax"];
                    y2 = bbox["ymax"];
                    wr = label + " " + x1 + " " + y1 + " " + x2 + " " + y2 + "\n";
                    f.write(wr);
                    
            else:
                obj = dict(anno["object"]);
                label = anno["object"]["name"];
                bbox = dict(anno["object"]["bndbox"])
                x1 = bbox["xmin"];
                y1 = bbox["ymin"];
                x2 = bbox["xmax"];
                y2 = bbox["ymax"];
                
                wr = label + " " + x1 + " " + y1 + " " + x2 + " " + y2 + "\n";
                f.write(wr);
                
            f.close();

        MINOVERLAP = iou_thresh # default value (defined in the PASCAL VOC2012 challenge)


        class Args():
            def __init__(self, verbose=1):
                self.no_animation = False;
                self.no_plot = False;
                self.quiet = False;
                self.ignore = "";
                self.set_class_iou = "";

        args = Args();

        '''
            0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)
        '''

        # if there are no classes to ignore then replace None by empty list
        if args.ignore is None:
            args.ignore = []

        specific_iou_flagged = False
        if args.set_class_iou is not None:
            specific_iou_flagged = True



        GT_PATH = "tmp/gt/"
        DR_PATH = "tmp/result/"
        # if there are no images then no animation can be shown
        args.no_animation = True

        # try to import OpenCV if the user didn't choose the option --no-animation
        show_animation = False
        if not args.no_animation:
            try:
                import cv2
                show_animation = True
            except ImportError:
                print("\"opencv-python\" not found, please install to visualize the results.")
                args.no_animation = True

        # try to import Matplotlib if the user didn't choose the option --no-plot
        draw_plot = True




        """
         Create a ".temp_files/" and "output/" directory
        """
        TEMP_FILES_PATH = ".temp_files"
        if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
            os.makedirs(TEMP_FILES_PATH)
        output_files_path = "evaluation_results"
        if os.path.exists(output_files_path): # if it exist already
            # reset the output directory
            shutil.rmtree(output_files_path)

        os.makedirs(output_files_path)
        if draw_plot:
            os.makedirs(os.path.join(output_files_path, "classes"))
        if show_animation:
            os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

        """
         ground-truth
             Load each of the ground-truth files into a temporary ".json" file.
             Create a list of all the class names present in the ground-truth (gt_classes).
        """
        # get a list with the ground-truth files
        ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
        if len(ground_truth_files_list) == 0:
            error("Error: No ground-truth files found!")
        ground_truth_files_list.sort()
        # dictionary with counter per class
        gt_counter_per_class = {}
        counter_images_per_class = {}

        gt_files = []
        for txt_file in ground_truth_files_list:
            #print(txt_file)
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # check if there is a correspondent detection-results file
            temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                error(error_msg)
            lines_list = file_lines_to_list(txt_file)
            # create ground-truth dictionary
            bounding_boxes = []
            is_difficult = False
            already_seen_classes = []
            for line in lines_list:
                try:
                    if "difficult" in line:
                            class_name, left, top, right, bottom, _difficult = line.split()
                            is_difficult = True
                    else:
                            class_name, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                    error_msg += " Received: " + line
                    error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                    error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                    error(error_msg)
                # check if class is in the ignore list, if yes skip
                if class_name in args.ignore:
                    continue
                bbox = left + " " + top + " " + right + " " +bottom
                if is_difficult:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                    # count that object
                    if class_name in gt_counter_per_class:
                        gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        gt_counter_per_class[class_name] = 1

                    if class_name not in already_seen_classes:
                        if class_name in counter_images_per_class:
                            counter_images_per_class[class_name] += 1
                        else:
                            # if class didn't exist yet
                            counter_images_per_class[class_name] = 1
                        already_seen_classes.append(class_name)


            # dump bounding_boxes into a ".json" file
            new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            gt_files.append(new_temp_file)
            with open(new_temp_file, 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        #print(gt_classes)
        #print(gt_counter_per_class)

        """
         Check format of the flag --set-class-iou (if used)
            e.g. check if class exists
        """
        if specific_iou_flagged:
            n_args = len(args.set_class_iou)
            error_msg = \
                '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
            if n_args % 2 != 0:
                error('Error, missing arguments. Flag usage:' + error_msg)
            # [class_1] [IoU_1] [class_2] [IoU_2]
            # specific_iou_classes = ['class_1', 'class_2']
            specific_iou_classes = args.set_class_iou[::2] # even
            # iou_list = ['IoU_1', 'IoU_2']
            iou_list = args.set_class_iou[1::2] # odd
            if len(specific_iou_classes) != len(iou_list):
                error('Error, missing arguments. Flag usage:' + error_msg)
            for tmp_class in specific_iou_classes:
                if tmp_class not in gt_classes:
                            error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
            for num in iou_list:
                if not is_float_between_0_and_1(num):
                    error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

        """
         detection-results
             Load each of the detection-results files into a temporary ".json" file.
        """
        # get a list with the detection-results files
        dr_files_list = glob.glob(DR_PATH + '/*.txt')
        dr_files_list.sort()

        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for txt_file in dr_files_list:
                #print(txt_file)
                # the first time it checks if all the corresponding ground-truth files exist
                file_id = txt_file.split(".txt",1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
                if class_index == 0:
                    if not os.path.exists(temp_path):
                        error_msg = "Error. File not found: {}\n".format(temp_path)
                        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                        error(error_msg)
                lines = file_lines_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                        error_msg += " Received: " + line
                        error(error_msg)
                    if tmp_class_name == class_name:
                        #print("match")
                        bbox = left + " " + top + " " + right + " " +bottom
                        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                        #print(bounding_boxes)
            # sort detection-results by decreasing confidence
            bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
            with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        """
         Calculate the AP for each class
        """
        sum_AP = 0.0
        ap_dictionary = {}
        lamr_dictionary = {}
        # open file to store the output
        with open(output_files_path + "/output.txt", 'w') as output_file:
            output_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                """
                 Load detection-results of that class
                """
                dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
                dr_data = json.load(open(dr_file))

                """
                 Assign detection-results to ground-truth objects
                """
                nd = len(dr_data)
                tp = [0] * nd # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]
                    if show_animation:
                        # find ground truth image
                        ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                        #tifCounter = len(glob.glob1(myPath,"*.tif"))
                        if len(ground_truth_img) == 0:
                            error("Error. Image not found with id: " + file_id)
                        elif len(ground_truth_img) > 1:
                            error("Error. Multiple image with id: " + file_id)
                        else: # found image
                            #print(IMG_PATH + "/" + ground_truth_img[0])
                            # Load image
                            img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                            # load image with draws of multiple detections
                            img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                            if os.path.isfile(img_cumulative_path):
                                img_cumulative = cv2.imread(img_cumulative_path)
                            else:
                                img_cumulative = img.copy()
                            # Add bottom border to image
                            bottom_border = 60
                            BLACK = [0, 0, 0]
                            img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                    # assign detection-results to ground truth object if any
                    # open ground-truth with that file_id
                    gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bb = [ float(x) for x in detection["bbox"].split() ]
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [ float(x) for x in obj["bbox"].split() ]
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    # assign detection as true positive/don't care/false positive
                    if show_animation:
                        status = "NO MATCH FOUND!" # status is only used in the animation
                    # set minimum overlap
                    min_overlap = MINOVERLAP
                    if specific_iou_flagged:
                        if class_name in specific_iou_classes:
                            index = specific_iou_classes.index(class_name)
                            min_overlap = float(iou_list[index])
                    if ovmax >= min_overlap:
                        if "difficult" not in gt_match:
                                if not bool(gt_match["used"]):
                                    # true positive
                                    tp[idx] = 1
                                    gt_match["used"] = True
                                    count_true_positives[class_name] += 1
                                    # update the ".json" file
                                    with open(gt_file, 'w') as f:
                                            f.write(json.dumps(ground_truth_data))
                                    if show_animation:
                                        status = "MATCH!"
                                else:
                                    # false positive (multiple detection)
                                    fp[idx] = 1
                                    if show_animation:
                                        status = "REPEATED MATCH!"
                    else:
                        # false positive
                        fp[idx] = 1
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"

                    """
                     Draw image to show animation
                    """
                    if show_animation:
                        height, widht = img.shape[:2]
                        # colors (OpenCV works with BGR)
                        white = (255,255,255)
                        light_blue = (255,200,100)
                        green = (0,255,0)
                        light_red = (30,30,255)
                        # 1st line
                        margin = 10
                        v_pos = int(height - margin - (bottom_border / 2.0))
                        text = "Image: " + ground_truth_img[0] + " "
                        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                        if ovmax != -1:
                            color = light_red
                            if status == "INSUFFICIENT OVERLAP":
                                text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                                color = green
                            img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                        # 2nd line
                        v_pos += int(bottom_border / 2.0)
                        rank_pos = str(idx+1) # rank position (idx starts at 0)
                        text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        color = light_red
                        if status == "MATCH!":
                            color = green
                        text = "Result: " + status + " "
                        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if ovmax > 0: # if there is intersections between the bounding-boxes
                            bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
                            cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                            cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                            cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                        bb = [int(i) for i in bb]
                        cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                        cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                        cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                        # show image
                        cv2.imshow("Animation", img)
                        cv2.waitKey(20) # show for 20 ms
                        # save image to output
                        output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                        cv2.imwrite(output_img_path, img)
                        # save the image with all the objects drawn to it
                        cv2.imwrite(img_cumulative_path, img_cumulative)

                #print(tp)
                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                #print(tp)
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                #print(rec)
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                #print(prec)

                ap, mrec, mprec = voc_ap(rec[:], prec[:])
                sum_AP += ap
                text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
                """
                 Write to output.txt
                """
                rounded_prec = [ '%.2f' % elem for elem in prec ]
                rounded_rec = [ '%.2f' % elem for elem in rec ]
                output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
                if not args.quiet:
                    print(text)
                ap_dictionary[class_name] = ap

                n_images = counter_images_per_class[class_name]
                lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
                lamr_dictionary[class_name] = lamr

                """
                 Draw plot
                """
                if draw_plot:
                    plt.plot(rec, prec, '-o')
                    # add a new penultimate point to the list (mrec[-2], 0.0)
                    # since the last line segment (and respective area) do not affect the AP value
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                    # set window title
                    fig = plt.gcf() # gcf - get current figure
                    fig.canvas.set_window_title('AP ' + class_name)
                    # set plot title
                    plt.title('class: ' + text)
                    #plt.suptitle('This is a somewhat long figure title', fontsize=16)
                    # set axis titles
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    # optional - set axes
                    axes = plt.gca() # gca - get current axes
                    axes.set_xlim([0.0,1.0])
                    axes.set_ylim([0.0,1.05]) # .05 to give some extra space
                    # Alternative option -> wait for button to be pressed
                    #while not plt.waitforbuttonpress(): pass # wait for key display
                    # Alternative option -> normal display
                    #plt.show()
                    # save the plot
                    fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                    plt.cla() # clear axes for next plot

            if show_animation:
                cv2.destroyAllWindows()

            output_file.write("\n# mAP of all classes\n")
            mAP = sum_AP / n_classes
            text = "mAP = {0:.2f}%".format(mAP*100)
            output_file.write(text + "\n")
            print(text)

        """
         Draw false negatives
        """
        if show_animation:
            pink = (203,192,255)
            for tmp_file in gt_files:
                ground_truth_data = json.load(open(tmp_file))
                #print(ground_truth_data)
                # get name of corresponding image
                start = TEMP_FILES_PATH + '/'
                img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
                img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
                img = cv2.imread(img_cumulative_path)
                if img is None:
                    img_path = IMG_PATH + '/' + img_id + ".jpg"
                    img = cv2.imread(img_path)
                # draw false negatives
                for obj in ground_truth_data:
                    if not obj['used']:
                        bbgt = [ int(round(float(x))) for x in obj["bbox"].split() ]
                        cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),pink,2)
                cv2.imwrite(img_cumulative_path, img)

        # remove the temp_files directory
        shutil.rmtree(TEMP_FILES_PATH)

        """
         Count total of detection-results
        """
        # iterate through all the files
        det_counter_per_class = {}
        for txt_file in dr_files_list:
            # get lines to list
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                if class_name in args.ignore:
                    continue
                # count that object
                if class_name in det_counter_per_class:
                    det_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    det_counter_per_class[class_name] = 1
        #print(det_counter_per_class)
        dr_classes = list(det_counter_per_class.keys())


        """
         Plot the total number of occurences of each class in the ground-truth
        """
        if draw_plot:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = output_files_path + "/ground-truth-info.png"
            to_show = False
            plot_color = 'forestgreen'
            draw_plot_func(
                gt_counter_per_class,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                '',
                )

        """
         Write number of ground-truth objects per class to results.txt
        """
        with open(output_files_path + "/output.txt", 'a') as output_file:
            output_file.write("\n# Number of ground-truth objects per class\n")
            for class_name in sorted(gt_counter_per_class):
                output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

        """
         Finish counting true positives
        """
        for class_name in dr_classes:
            # if class exists in detection-result but not in ground-truth then there are no true positives in that class
            if class_name not in gt_classes:
                count_true_positives[class_name] = 0
        #print(count_true_positives)

        """
         Plot the total number of occurences of each class in the "detection-results" folder
        """
        if draw_plot:
            window_title = "detection-results-info"
            # Plot title
            plot_title = "detection-results\n"
            plot_title += "(" + str(len(dr_files_list)) + " files and "
            count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            # end Plot title
            x_label = "Number of objects per class"
            output_path = output_files_path + "/detection-results-info.png"
            to_show = False
            plot_color = 'forestgreen'
            true_p_bar = count_true_positives
            draw_plot_func(
                det_counter_per_class,
                len(det_counter_per_class),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                true_p_bar
                )

        """
         Write number of detected objects per class to output.txt
        """
        with open(output_files_path + "/output.txt", 'a') as output_file:
            output_file.write("\n# Number of detected objects per class\n")
            for class_name in sorted(dr_classes):
                n_det = det_counter_per_class[class_name]
                text = class_name + ": " + str(n_det)
                text += " (tp:" + str(count_true_positives[class_name]) + ""
                text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
                output_file.write(text)

        """
         Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
        """
        if draw_plot:
            window_title = "lamr"
            plot_title = "log-average miss rate"
            x_label = "log-average miss rate"
            output_path = output_files_path + "/lamr.png"
            to_show = False
            plot_color = 'royalblue'
            draw_plot_func(
                lamr_dictionary,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
                )

        """
         Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if draw_plot:
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mAP*100)
            x_label = "Average Precision"
            output_path = output_files_path + "/mAP.png"
            to_show = False
            plot_color = 'royalblue'
            draw_plot_func(
                ap_dictionary,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
                )

        if(os.path.isdir("tmp")):
            os.system("rm -r tmp");

        return mAP, ap_dictionary, lamr_dictionary;





    
    
    def benchmark_for_speed(self, img_path, img_size=300, thresh=0.5, bbox_thickness=3, text_size=2, text_thickness=4):
        img = cv.imread(img_path)
        rows = img.shape[0]
        cols = img.shape[1]
        if(img_size):
            inp = cv.resize(img, (img_size, img_size))
        else:
            inp = img;
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = self.system_dict["sess"].run([self.system_dict["sess"].graph.get_tensor_by_name('num_detections:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_scores:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_boxes:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})             
              
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        # Iterate through all detected detections

        scores = [];
        bboxes = [];
        labels = [];
        for i in range(num_detections):
            classId = int(out[3][0][i])

            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            label = self.system_dict["classes"][int(out[3][0][i])-1]

            if score > thresh:
                # Creating a box around the detected number plate
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                cv.rectangle(img, (x, y), (right, bottom), (0, 0, 255), thickness=bbox_thickness)
                cv.putText(img, label, (x,y), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 225), text_thickness)
                cv.imwrite('output.png', img)
                scores.append(score);
                bboxes.append([x, y, right, bottom])
                labels.append(label);
       
        img_processing_time = 0.0;
        inference_time = 0.0;
        result_extraction_time = 0.0;
        iter_times = [];
        
         
        for _ in range(100):
            start = time.time();
            img = cv.imread(img_path)
            rows = img.shape[0]
            cols = img.shape[1]
            if(img_size):
                inp = cv.resize(img, (img_size, img_size))
            else:
                inp = img;
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            end = time.time();
            img_processing_time += end - start;
            
            start = time.time();
            out = self.system_dict["sess"].run([self.system_dict["sess"].graph.get_tensor_by_name('num_detections:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_scores:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_boxes:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            end = time.time();
            inference_time += end - start;
            iter_times.append(end-start);
            
            start = time.time();
            num_detections = int(out[0][0])
            scores = [];
            bboxes = [];
            labels = [];
            for i in range(num_detections):
                classId = int(out[3][0][i])

                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                label = self.system_dict["classes"][int(out[3][0][i])-1]

                if score > thresh:
                    # Creating a box around the detected number plate
                    x = int(bbox[1] * cols)
                    y = int(bbox[0] * rows)
                    right = int(bbox[3] * cols)
                    bottom = int(bbox[2] * rows)
                    cv.rectangle(img, (x, y), (right, bottom), (0, 0, 255), thickness=bbox_thickness)
                    cv.putText(img, label, (x,y), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 225), text_thickness)
                    cv.imwrite('output.png', img)
                    scores.append(score);
                    bboxes.append([x, y, right, bottom])
                    labels.append(label);
            end = time.time();
            result_extraction_time += end - start;
            
            
        print("Average Image loading time: {}".format(img_processing_time/100));
        print("Average Inference time: {}".format(inference_time/100));
        print("Result extraction time: {}".format(result_extraction_time/100));
        
        iter_times = np.array(iter_times)
        print('total_time = {}'.format(np.sum(iter_times)))
        print('images_per_sec = {}'.format(int(np.mean(1 / iter_times))))
        print('99th_percentile = {}'.format(np.percentile(iter_times, q=99, interpolation='lower') * 1000))
        print('latency_mean  = {}'.format(np.mean(iter_times) * 1000))
        print('latency_median = {}'.format(np.median(iter_times) * 1000))
        print('latency_min = {}'.format(np.min(iter_times) * 1000))
            
            
            
        
        
        
        