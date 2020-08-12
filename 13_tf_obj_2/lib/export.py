import os
import json
from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

tf.enable_v2_behavior()

def main(_):
    with open('system_dict.json') as json_file:
        args = json.load(json_file) 
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(args["pipeline_config_path"], 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(args["config_override"], pipeline_config)
    exporter_lib_v2.export_inference_graph(
                args["input_type"], pipeline_config, args["trained_checkpoint_dir"],
                args["output_directory"])
    
    
app.run(main=main)