import os
import sys
sys.path.append("Monk_Object_Detection/12_tf_obj_1/lib/")

from infer_detector_nano import Infer

gtf = Infer();

# Model loading takes time on nano boards
gtf.set_model_params('trt_fp16_dir/trt_graph.pb', "ship/classes.txt")

# Running for the first time builds the tensorRT engine for the model based on the plan saved in trt_fp16_dir folder
# Oputput will be saved as output.png
scores, bboxes, labels = gtf.infer_on_image('ship/test/img5.jpg', thresh=0.5, img_size=300);

# Run speed benchmark
gtf.benchmark_for_speed('ship/test/img1.jpg', img_size=300)