import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import time
from os import listdir
from os.path import isfile, join


from tensorflow.python.tools import freeze_graph

class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        
    
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
        
        
    def infer_on_image(self, img_path, img_size=300, thresh=0.5, bbox_thickness=3, text_size=2, text_thickness=4):
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
        print("Image loaded in {} sec".format(end-start));

        # Run the model
        start = time.time();
        out = self.system_dict["sess"].run([self.system_dict["sess"].graph.get_tensor_by_name('num_detections:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_scores:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_boxes:0'),
                        self.system_dict["sess"].graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        end = time.time();
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
        print("Inference printed on image in {} sec".format(end-start));
        
        return scores, bboxes, labels;
        
        
        
        