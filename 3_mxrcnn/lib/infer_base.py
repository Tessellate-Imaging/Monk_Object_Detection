import os
import sys

sys.path.append("mx-rcnn");

import ast
import pprint

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"; 
import mxnet as mx
from mxnet.module import Module

from symdata.bbox import im_detect
from symdata.loader import load_test, generate_batch
from symdata.vis import vis_detection, save_detection
from symnet.model import load_param, check_shape

system_dict = {};


#######################################################################################################################################
def set_class_list(class_list_file):
    '''
    User function: Get class list from file

    Args:
        class_list_file (str): Path to file containing all class names

    Returns:
        None
    '''
    f = open(class_list_file, 'r');
    system_dict["classes"] = ["__background__"];
    system_dict["classes"] += f.readlines();
    f.close();
    system_dict["rcnn_num_classes"] = len(system_dict["classes"]);
#######################################################################################################################################




#######################################################################################################################################
def set_model_params(model_name="vgg16", model_path=None):
    '''
    User function: Set model parameters

        Available models
            vgg16
            resnet50
            resnet101

    Args:
        model_name (str): Select from available models
        model_path (str): Path to model file

    Returns:
        None
    '''
    system_dict["network"] = model_name;
    system_dict["params"] = model_path;
    if(model_name == "vgg16"):
        system_dict["rcnn_feat_stride"] = 16;
        system_dict["rcnn_pooled_size"] = '(7, 7)';
        system_dict["net_fixed_params"] = '["conv1", "conv2", "conv3", "conv4"]';
    elif(model_name == "resnet50" or model_name == "resnet101"):
        system_dict["rcnn_feat_stride"] = 16;
        system_dict["rcnn_pooled_size"] = '(14, 14)';
        system_dict["net_fixed_params"] = '["conv0", "stage1", "gamma", "beta"]';

    system_dict["net_fixed_params"] = ast.literal_eval(system_dict["net_fixed_params"])

#######################################################################################################################################



#######################################################################################################################################
def set_img_preproc_params(img_short_side=600, img_long_side=1000, mean=(123.68, 116.779, 103.939), std=(1.0, 1.0, 1.0)):
    '''
    User function: Set image preprocessing parameters

    Args:
        img_short_side (int): Minimum image size for rescaling
        img_long_side (int): Maximum image size for rescaling
        mean (tuple): 3-Channel mean for subtraction in preprocessing
        std (tuple): 3-Channel standard deviation for normalizing in preprocessing

    Returns:
        None
    '''
    system_dict["img_short_side"] = img_short_side;
    system_dict["img_long_side"] = img_long_side;
    system_dict["img_pixel_means"] = str(mean);
    system_dict["img_pixel_stds"] = str(std);
    system_dict["img_pixel_means"] = ast.literal_eval(system_dict["img_pixel_means"])
    system_dict["img_pixel_stds"] = ast.literal_eval(system_dict["img_pixel_stds"])
#######################################################################################################################################




#######################################################################################################################################
def initialize_rpn_params():
    '''
    User function: Initialize all RPN parameters

    Args:
        None

    Returns:
        None
    '''
    system_dict["rpn_feat_stride"] = 16;
    system_dict["rpn_anchor_scales"] = '(8, 16, 32)';
    system_dict["rpn_anchor_ratios"] = '(0.5, 1, 2)';
    system_dict["rpn_pre_nms_topk"] = 12000;
    system_dict["rpn_post_nms_topk"] = 2000;
    system_dict["rpn_nms_thresh"] = 0.7;
    system_dict["rpn_min_size"] = 16;
    system_dict["rpn_batch_rois"] = 256;
    system_dict["rpn_allowed_border"] = 0;
    system_dict["rpn_fg_fraction"] = 0.5;
    system_dict["rpn_fg_overlap"] = 0.7;
    system_dict["rpn_bg_overlap"] = 0.3;
    
    system_dict["rpn_anchor_scales"] = ast.literal_eval(system_dict["rpn_anchor_scales"])
    system_dict["rpn_anchor_ratios"] = ast.literal_eval(system_dict["rpn_anchor_ratios"])


def initialize_rcnn_params():
    '''
    User function: Initialize all RCNN parameters

    Args:
        None

    Returns:
        None
    '''
    system_dict["rcnn_batch_rois"] = 128;
    system_dict["rcnn_fg_fraction"] = 0.25;
    system_dict["rcnn_fg_overlap"] = 0.5;
    system_dict["rcnn_bbox_stds"] = '(0.1, 0.1, 0.2, 0.2)';
    system_dict["rcnn_nms_thresh"] = 0.3;
    system_dict["rcnn_conf_thresh"] = 0.001;

    system_dict["rcnn_pooled_size"] = ast.literal_eval(system_dict["rcnn_pooled_size"])
    system_dict["rcnn_bbox_stds"] = ast.literal_eval(system_dict["rcnn_bbox_stds"])
#######################################################################################################################################



#######################################################################################################################################
def set_hyper_params(gpus="0", batch_size=1):
    '''
    User function: Set hyper parameters

    Args:
        gpus (string): String mentioning gpu device ID to run the inference on.

    Returns:
        None
    '''
    system_dict["gpu"] = gpus.split(",")[0];
    system_dict["rcnn_batch_size"] = batch_size;
#######################################################################################################################################



#######################################################################################################################################
def set_output_params(vis_thresh=0.8, vis=False):
    '''
    User function: Set output parameters

    Args:
        vis_thresh (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
        vis (bool): If True, the output will be displayed.

    Returns:
        None
    '''
    system_dict["vis_thresh"] = vis_thresh;
    system_dict["vis"] = vis;

#######################################################################################################################################




#######################################################################################################################################
def get_vgg16_test(system_dict):
    '''
    Internal function: Select vgg16 params

    Args:
        system_dict (dict): Dictionary of all the parameters selected for training

    Returns:
        mxnet model: Vgg16 model
    '''
    from symnet.symbol_vgg import get_vgg_test

    return get_vgg_test(anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"],
                        rpn_feature_stride=system_dict["rpn_feat_stride"], rpn_pre_topk=system_dict["rpn_pre_nms_topk"],
                        rpn_post_topk=system_dict["rpn_post_nms_topk"], rpn_nms_thresh=system_dict["rpn_nms_thresh"],
                        rpn_min_size=system_dict["rpn_min_size"],
                        num_classes=system_dict["rcnn_num_classes"], rcnn_feature_stride=system_dict["rcnn_feat_stride"],
                        rcnn_pooled_size=system_dict["rcnn_pooled_size"], rcnn_batch_size=system_dict["rcnn_batch_size"])


def get_resnet50_test(system_dict):
    '''
    Internal function: Select resnet50 params

    Args:
        system_dict (dict): Dictionary of all the parameters selected for training

    Returns:
        mxnet model: Resnet50 model
    '''
    from symnet.symbol_resnet import get_resnet_test

    return get_resnet_test(anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"],
                           rpn_feature_stride=system_dict["rpn_feat_stride"], rpn_pre_topk=system_dict["rpn_pre_nms_topk"],
                           rpn_post_topk=system_dict["rpn_post_nms_topk"], rpn_nms_thresh=system_dict["rpn_nms_thresh"],
                           rpn_min_size=system_dict["rpn_min_size"],
                           num_classes=system_dict["rcnn_num_classes"], rcnn_feature_stride=system_dict["rcnn_feat_stride"],
                           rcnn_pooled_size=system_dict["rcnn_pooled_size"], rcnn_batch_size=system_dict["rcnn_batch_size"],
                           units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


def get_resnet101_test(system_dict):
    '''
    Internal function: Select resnet101 params

    Args:
        system_dict (dict): Dictionary of all the parameters selected for training

    Returns:
        mxnet model: Resnet101 model
    '''
    from symnet.symbol_resnet import get_resnet_test

    return get_resnet_test(anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"],
                           rpn_feature_stride=system_dict["rpn_feat_stride"], rpn_pre_topk=system_dict["rpn_pre_nms_topk"],
                           rpn_post_topk=system_dict["rpn_post_nms_topk"], rpn_nms_thresh=system_dict["rpn_nms_thresh"],
                           rpn_min_size=system_dict["rpn_min_size"],
                           num_classes=system_dict["rcnn_num_classes"], rcnn_feature_stride=system_dict["rcnn_feat_stride"],
                           rcnn_pooled_size=system_dict["rcnn_pooled_size"], rcnn_batch_size=system_dict["rcnn_batch_size"],
                           units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))






def set_network():
    '''
    User function: Set the train model

    Args:
        None

    Returns:
        mxnet model: Model as per selected params
    '''
    network = system_dict["network"]
    networks = {
        'vgg16': get_vgg16_test,
        'resnet50': get_resnet50_test,
        'resnet101': get_resnet101_test
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    
    return networks[network](system_dict)
#######################################################################################################################################






def load_model(sym):
    '''
    User function: Loads the trained model weights 

    Args:
        sym (mxnet model): Mxnet model returned from set_network() function

    Returns:
        mxnet model: Model with trained weights
    '''
    if system_dict["gpu"]:
        ctx = mx.gpu(int(system_dict["gpu"]))
    else:
        ctx = mx.cpu(0)

    # load params
    arg_params, aux_params = load_param(system_dict["params"], ctx=ctx)

    # produce shape max possible
    data_names = ['data', 'im_info']
    label_names = None
    data_shapes = [('data', (1, 3, system_dict["img_long_side"], system_dict["img_long_side"])), ('im_info', (1, 3))]
    label_shapes = None

    # check shapes
    check_shape(sym, data_shapes, arg_params, aux_params)

    # create and bind module
    mod = Module(sym, data_names, label_names, context=ctx)
    mod.bind(data_shapes, label_shapes, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    return mod;







#######################################################################################################################################
def Infer(img_name, mod):
    '''
    User function: Run inference on image and visualize it

    Args:
        img_name (str): Relative path to the image file
        mod (mxnet model): Mxnet model returned from load_model() function

    Returns:
        list: Contaning IDs, Scores and bounding box locations of predicted objects. 
    '''
    system_dict["image"] = img_name;
    if system_dict["gpu"]:
        ctx = mx.gpu(int(system_dict["gpu"]))
    else:
        ctx = mx.cpu(0)

    # load single test
    im_tensor, im_info, im_orig = load_test(system_dict["image"], short=system_dict["img_short_side"], 
                                            max_size=system_dict["img_long_side"],
                                            mean=system_dict["img_pixel_means"], std=system_dict["img_pixel_stds"])


    # generate data batch
    data_batch = generate_batch(im_tensor, im_info)


    # forward
    mod.forward(data_batch)
    rois, scores, bbox_deltas = mod.get_outputs()
    rois = rois[:, 1:]
    scores = scores[0]
    bbox_deltas = bbox_deltas[0]
    im_info = im_info[0]


    # decode detection
    det = im_detect(rois, scores, bbox_deltas, im_info,
                    bbox_stds=system_dict["rcnn_bbox_stds"], nms_thresh=system_dict["rcnn_nms_thresh"],
                    conf_thresh=system_dict["rcnn_conf_thresh"])


    output = [];
    conf_scores = [];
    for [cls, conf, x1, y1, x2, y2] in det:
        output.append([system_dict["classes"][int(cls)], conf, [x1, y1, x2, y2]]);
        conf_scores.append(conf)
        if cls > 0 and conf > system_dict["vis_thresh"]:
            print(system_dict["classes"][int(cls)], conf, [x1, y1, x2, y2])
    
    
    max_index = conf_scores.index(max(conf_scores))
    print(output[max_index])
    
    if system_dict["vis"]:
        vis_detection(im_orig, det, system_dict["classes"], thresh=system_dict["vis_thresh"])
    
    save_detection(im_orig, det, system_dict["classes"], thresh=system_dict["vis_thresh"])

    return output;



#######################################################################################################################################
