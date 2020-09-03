import os
import tensorflow as tf
import json
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

with open('system_dict.json') as json_file:
    args = json.load(json_file) 

input_saved_model_dir = args["output_directory"] + "/saved_model";
input_saved_model_dir = input_saved_model_dir.replace("//", "/");
trt_dir = args["trt_dir"] + "/saved_model";
trt_dir = trt_dir.replace("//", "/");
conversion_type = args["conversion_type"]
image_size = int(args["input_shape"].split(",")[1])


def input_fn():
    global image_size;
    print("image_size = ", image_size);
    for i in range(10):
        yield tf.cast(tf.random.uniform((1, 1, image_size, image_size, 3)), tf.dtypes.uint8)



if(conversion_type == "FP16" or conversion_type == "FP32"):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode=conversion_type)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, 
                                        conversion_params=conversion_params)
    converter.convert()
    converter.save(trt_dir)
elif(conversion_type == "INT8"):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(precision_mode="INT8", 
                                                   maximum_cached_engines=1, 
                                                   use_calibration=True)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, 
                                        conversion_params=conversion_params)

    converter.convert(calibration_input_fn=input_fn)
    converter.save(trt_dir)

cmd = "cp " + args["output_directory"] + "/pipeline.config " + args["trt_dir"] + "/";
os.system(cmd);