import os
import sys
import json
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection import exporter_flops
from object_detection.protos import pipeline_pb2

#python models/research/object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=ssd_mobilenet_v1_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix=output_dir/model.ckpt-10000 --output_directory=output_export_dir


def main(_):
    with open('system_dict.json') as json_file:
        args = json.load(json_file) 
    
    
    if(os.path.isdir(args["output_directory"])):
        os.system("rm -r " + args["output_directory"]);
    
    os.mkdir(args["output_directory"]);
       
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(args["pipeline_config_path"], 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(args["config_override"], pipeline_config)
    
    if args["input_shape"]:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in args["input_shape"].split(',')
        ]
    else:
        input_shape = None
        
    if args["input_shape_flops"]:
        input_shape_flops = [
            int(dim) if dim != '-1' else None
            for dim in args["input_shape_flops"].split(',')
        ]
    else:
        input_shape_flops = None
    
    if args["use_side_inputs"]:
        side_input_shapes, side_input_names, side_input_types = (
            exporter.parse_side_inputs(
                args["side_input_shapes"],
                args["side_input_names"],
                args["side_input_types"]))
    else:
        side_input_shapes = None
        side_input_names = None
        side_input_types = None
        
    
    if args["additional_output_tensor_names"]:
        additional_output_tensor_names = list(
            args["additional_output_tensor_names"].split(','))
    else:
        additional_output_tensor_names = None

    exporter_flops.export_inference_graph(
                args["input_type"], 
                pipeline_config, 
                args["trained_checkpoint_prefix"],
                args["output_directory"] + "_flops", 
                input_shape=input_shape_flops,
                write_inference_graph=args["write_inference_graph"],
                additional_output_tensor_names=additional_output_tensor_names,
                use_side_inputs=args["use_side_inputs"],
                side_input_shapes=side_input_shapes,
                side_input_names=side_input_names,
                side_input_types=side_input_types)
    
    
tf.app.run(main=main)
    
    
    
    
    
    
    
    
    
    
    
    
