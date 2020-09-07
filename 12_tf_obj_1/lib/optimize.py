import tensorflow.contrib.tensorrt as trt

from object_detection.protos import pipeline_pb2
from object_detection.protos import image_resizer_pb2
from object_detection import exporter

import os
import json
import subprocess

from collections import namedtuple
from google.protobuf import text_format

import tensorflow as tf


def make_const6(const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
    return graph.as_graph_def()


def make_relu6(output_name, input_name, const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_x = tf.placeholder(tf.float32, [10, 10], name=input_name)
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
        with tf.name_scope(output_name):
            tf_y1 = tf.nn.relu(tf_x, name='relu1')
            tf_y2 = tf.nn.relu(tf.subtract(tf_x, tf_6, name='sub1'), name='relu2')

            #tf_y = tf.nn.relu(tf.subtract(tf_6, tf.nn.relu(tf_x, name='relu1'), name='sub'), name='relu2')
        #tf_y = tf.subtract(tf_6, tf_y, name=output_name)
        tf_y = tf.subtract(tf_y1, tf_y2, name=output_name)
        
    graph_def = graph.as_graph_def()
    graph_def.node[-1].name = output_name

    # remove unused nodes
    for node in graph_def.node:
        if node.name == input_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.name == const6_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.op == '_Neg':
            node.op = 'Neg'
            
    return graph_def


def convert_relu6(graph_def, const6_name='const6'):
    # add constant 6
    has_const6 = False
    for node in graph_def.node:
        if node.name == const6_name:
            has_const6 = True
    if not has_const6:
        const6_graph_def = make_const6(const6_name=const6_name)
        graph_def.node.extend(const6_graph_def.node)
        
    for node in graph_def.node:
        if node.op == 'Relu6':
            input_name = node.input[0]
            output_name = node.name
            relu6_graph_def = make_relu6(output_name, input_name, const6_name=const6_name)
            graph_def.node.remove(node)
            graph_def.node.extend(relu6_graph_def.node)
            
    return graph_def


def remove_node(graph_def, node):
    for n in graph_def.node:
        if node.name in n.input:
            n.input.remove(node.name)
        ctrl_name = '^' + node.name
        if ctrl_name in n.input:
            n.input.remove(ctrl_name)
    graph_def.node.remove(node)


def remove_op(graph_def, op_name):
    matches = [node for node in graph_def.node if node.op == op_name]
    for match in matches:
        remove_node(graph_def, match)


def f_force_nms_cpu(frozen_graph):
    for node in frozen_graph.node:
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    return frozen_graph


def f_replace_relu6(frozen_graph):
    return convert_relu6(frozen_graph)


def f_remove_assert(frozen_graph):
    remove_op(frozen_graph, 'Assert')
    return frozen_graph



DetectionModel = namedtuple('DetectionModel', ['name', 'url', 'extract_dir'])

INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
MASKS_NAME='detection_masks'
NUM_DETECTIONS_NAME='num_detections'
FROZEN_GRAPH_NAME='frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME='pipeline.config'
CHECKPOINT_PREFIX='model.ckpt'


def get_input_names(model):
    return [INPUT_NAME]


def get_output_names(model):
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
    if model == 'mask_rcnn_resnet50_atrous_coco':
        output_names.append(MASKS_NAME)
    return output_names


def build_detection_graph(config, checkpoint,
        batch_size=1,
        score_threshold=None,
        force_nms_cpu=True,
        replace_relu6=True,
        remove_assert=True,
        input_shape=None,
        output_dir='.generated_model'):
    """Builds a frozen graph for a pre-trained object detection model"""
    
    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])
    os.mkdir(output_dir)

    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            exporter.export_inference_graph(
                'image_tensor', 
                config, 
                checkpoint_path, 
                output_dir, 
                input_shape=[batch_size, None, None, 3]
            )

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    # TODO: handle mask_rcnn 
    input_names = [INPUT_NAME]
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # remove temporary directory
    subprocess.call(['rm', '-rf', output_dir])

    return frozen_graph, input_names, output_names;

with open('system_dict.json') as json_file:
    args = json.load(json_file) 

precision_mode = args["conversion_type"];
trt_dir = args["trt_dir"];
config_path = args["pipeline_config_path"];
checkpoint_path = args["trained_checkpoint_prefix"]
frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path
)



if(precision_mode == "FP16" or precision_mode == "FP32"):
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode=precision_mode,
        minimum_segment_size=50,
        is_dynamic_op=True
    )
else:
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='INT8',
        minimum_segment_size=50,
        is_dynamic_op=False
    )


if(os.path.isdir(trt_dir)):
    os.system("rm -r " + trt_dir);
os.mkdir(trt_dir)
with open(trt_dir + '/trt_graph.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())
    
print("Optimized graph saved as {}".format(trt_dir + '/trt_graph.pb'));
print("Done");