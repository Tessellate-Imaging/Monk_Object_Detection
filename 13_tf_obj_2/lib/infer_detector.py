import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.saved_model import tag_constants

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils




import time


class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["model_type"] = "checkpoint"; #one of checkpoint, saved_model
    
    def load_image_into_numpy_array(self, path):
        return np.array(Image.open(path))

    def set_model_params(self, exported_model_dir="export_dir", model_type="saved_model"):
        saved_model_dir = exported_model_dir + '/saved_model';
        self.system_dict["saved_model_loaded"] = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
        self.system_dict["signature_keys"] = list(self.system_dict["saved_model_loaded"].signatures.keys())
        
        
        pipeline_config = exported_model_dir + '/pipeline.config';
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        
        label_map_path = configs['eval_input_config'].label_map_path
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.system_dict["category_index"] = label_map_util.create_category_index(categories)
        self.system_dict["label_map_dict"] = label_map_util.get_label_map_dict(label_map, use_display_name=True)
        
        
        
        
    def infer_on_image(self, image_path, thresh=0.5):
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
        
        
        start = time.time();
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              self.system_dict["category_index"],
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=thresh,
              agnostic_mode=False)

        output = Image.fromarray(np.uint8(image_np_with_detections)).convert('RGB')
        output = output.save("output.jpg") 
        
        end = time.time();
        print("Extracting results and priting on image time - {}".format(end-start));
              
        return detections['detection_scores'], detections['detection_boxes'], detections['detection_classes']
              
    
    def benchmark_for_speed(self, image_path):
        graph_func = self.system_dict["saved_model_loaded"].signatures[self.system_dict["signature_keys"][0]]
        img_processing_time = 0.0;
        inference_time = 0.0;
        result_extraction_time = 0.0;
        iter_times = [];
        

        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = graph_func(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              self.system_dict["category_index"],
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.30,
              agnostic_mode=False)

        output = Image.fromarray(np.uint8(image_np_with_detections)).convert('RGB')
        output = output.save("output.jpg") 
        
        
        
        
        
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


            start = time.time();
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                           for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()

            image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
                  image_np_with_detections,
                  detections['detection_boxes'],
                  detections['detection_classes'],
                  detections['detection_scores'],
                  self.system_dict["category_index"],
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=200,
                  min_score_thresh=.30,
                  agnostic_mode=False)

            output = Image.fromarray(np.uint8(image_np_with_detections)).convert('RGB')
            output = output.save("output.jpg") 

            end = time.time();
            result_extraction_time += end-start;
            
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
    
    
    
    
    
    '''
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
    
    '''
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    