import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.saved_model import tag_constants

class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        
            
    def set_dataset_params(self, class_list_file=None):
        f = open(class_list_file);
        lines = f.readlines();
        f.close();
        self.system_dict["classes"] = [];
        for i in range(len(lines)):
            if(lines[i] != ""):
                self.system_dict["classes"].append(lines[i][:len(lines[i])-1])
    
    def load_image_into_numpy_array(self, path):
        return np.array(Image.open(path))

    def set_model_params(self, exported_model_dir="export_dir"):
        saved_model_dir = exported_model_dir + '/saved_model';
        self.system_dict["saved_model_loaded"] = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
        self.system_dict["signature_keys"] = list(self.system_dict["saved_model_loaded"].signatures.keys())
        
    def infer_on_image(self, image_path, thresh=0.5):
        self.system_dict["image_path"] = image_path;
        self.system_dict["thresh"] = thresh;
                
        graph_func = self.system_dict["saved_model_loaded"].signatures[self.system_dict["signature_keys"][0]]
               
        start = time.time();
        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        end = time.time();
        print("Image loading and preproc time - {}".format(end-start));
        
        start = time.time();
        detections = graph_func(input_tensor)
        end = time.time();
        print("Inference time - {}".format(end-start));
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        self.system_dict["labels"] = detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        self.system_dict["boxes"] = detections['detection_boxes']
        self.system_dict["scores"] = detections['detection_scores']          
        
        return self.system_dict["scores"], self.system_dict["boxes"], self.system_dict["labels"]
        
        
    def draw_on_image(self, bbox_thickness=3, text_size=1, text_thickness=2):
        import cv2
        image_path = self.system_dict["image_path"];
        img = cv2.imread(image_path);
        h, w, c = img.shape
        thresh = self.system_dict["thresh"];
        
        labels = self.system_dict["labels"];
        scores = self.system_dict["scores"];
        boxes = self.system_dict["boxes"];
        
        for i in range(len(labels)):
            label = self.system_dict["classes"][labels[i]-1];
            score = scores[i];
            box = boxes[i];

            if(score > thresh):
                x1 = int(box[1]*w)
                y1 = int(box[0]*h)
                x2 = int(box[3]*w)
                y2 = int(box[2]*h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=bbox_thickness)
                cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 225), text_thickness)

        cv2.imwrite('output.jpg', img)   


    def benchmark_for_speed(self, image_path):
        graph_func = self.system_dict["saved_model_loaded"].signatures[self.system_dict["signature_keys"][0]]
        img_processing_time = 0.0;
        inference_time = 0.0;
        iter_times = [];
               
        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = graph_func(input_tensor)

        for i in range(100):
            start = time.time();
            image_np = self.load_image_into_numpy_array(image_path)
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]
            end = time.time();
            img_processing_time += end-start;

            start = time.time();
            detections = graph_func(input_tensor)
            end = time.time();
            inference_time += end-start;
            iter_times.append(end-start);

        print("Average Image loading time: {}".format(img_processing_time/100));
        print("Average Inference time: {}".format(inference_time/100));
        
        iter_times = np.array(iter_times)
        print('total_time = {}'.format(np.sum(iter_times)))
        print('images_per_sec = {}'.format(int(np.mean(1 / iter_times))))
        print('99th_percentile = {}'.format(np.percentile(iter_times, q=99, interpolation='lower') * 1000))
        print('latency_mean  = {}'.format(np.mean(iter_times) * 1000))
        print('latency_median = {}'.format(np.median(iter_times) * 1000))
        print('latency_min = {}'.format(np.min(iter_times) * 1000))

        
        
        
        
        
        
        
        
        