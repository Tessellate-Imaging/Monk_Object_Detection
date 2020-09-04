import os
import sys
sys.path.append("Monk_Object_Detection/13_tf_obj_2/lib/")

from infer_detector_nano import Infer

gtf = Infer();

gtf.set_dataset_params(class_list_file = 'ship/classes.txt')

gtf.set_model_params(exported_model_dir = 'trt_dir_int')

scores, boxes, labels = gtf.infer_on_image('ship/test/img1.jpg', thresh=0.5)

# optional
#gtf.draw_on_image(bbox_thickness=3, text_size=1, text_thickness=2)

gtf.benchmark_for_speed('ship/test/img1.jpg')