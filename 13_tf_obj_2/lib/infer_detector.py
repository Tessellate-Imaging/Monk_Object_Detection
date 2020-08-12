import os
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import cv2

class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        
    def load_image_into_numpy_array(self, path):
        image = cv2.imread(path);
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width, c = image.shape
        return np.array(image).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_keypoint_tuples(self, eval_config):
        tuple_list = []
        kp_list = eval_config.keypoint_edge
        for edge in kp_list:
            tuple_list.append((edge.start, edge.end))
        return tuple_list
    
    def get_model_detection_function(self, model):
        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn
    
    def set_model_params(self, exported_model_dir="export_dir"):
        pipeline_config = exported_model_dir + "/pipeline.config";
        model_dir = exported_model_dir + '/checkpoint/'
        
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(
              model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(
              model=detection_model)
        ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
        
        self.system_dict["detect_fn"] = self.get_model_detection_function(detection_model)
        
        label_map_path = configs['eval_input_config'].label_map_path
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.system_dict["category_index"] = label_map_util.create_category_index(categories)
        self.system_dict["label_map_dict"] = label_map_util.get_label_map_dict(label_map, use_display_name=True)
        
        
    def infer_on_image(self, image_path, img_size=300, thresh=0.5):
        import time
        start = time.time();
        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        end = time.time();
        print("Image loading time - ", end-start);

        start = time.time();
        detections, predictions_dict, shapes = self.system_dict["detect_fn"](input_tensor)
        end = time.time();
        print("Detection time - ", end - start);   
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        start = time.time();
        bboxes = detections['detection_boxes'][0].numpy()
        classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
        scores = detections['detection_scores'][0].numpy();
        
        image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
                                        image_np_with_detections,
                                        bboxes,
                                        classes,
                                        scores,
                                        self.system_dict["category_index"],
                                        use_normalized_coordinates=True,
                                        max_boxes_to_draw=200,
                                        min_score_thresh=thresh,
                                        agnostic_mode=False)
        cv2.imwrite("output.png", image_np_with_detections)
        end = time.time();
        print("Printing boxes and saving image time - ", end - start);
        
        
        return scores, bboxes, classes;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    