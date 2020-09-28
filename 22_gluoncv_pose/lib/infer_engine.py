from __future__ import division

import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler, LRSequential
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform
from gluoncv.utils.metrics import HeatmapAccuracy

import cv2
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord


class Infer():
    '''
    Class for main inference
    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};

    def Dataset_Params(self, num_joints=17):
        self.system_dict["local"]["num_joints"] = num_joints;

    def Model_Params(self, model_name="simple_pose_resnet18_v1b", params_file=None, use_gpu=True, detector_model_name='yolo3_mobilenet1.0_coco'):
        self.system_dict["local"]["params_file"] = params_file;
        self.system_dict["local"]["use_gpu"] = use_gpu;
        
        num_gpus = 1
        self.system_dict["local"]["ctx"] = [mx.gpu(i) for i in range(num_gpus)] if use_gpu > 0 else [mx.cpu()]

        self.system_dict["local"]["detector"] = model_zoo.get_model(detector_model_name, pretrained=True, ctx=self.system_dict["local"]["ctx"])

        self.system_dict["local"]["detector"].reset_class(["person"], reuse_weights=['person'])


        kwargs = {'ctx': self.system_dict["local"]["ctx"], 
                                        'num_joints': self.system_dict["local"]["num_joints"],
                                        'pretrained': True,
                                        'pretrained_base': True,
                                        'pretrained_ctx': self.system_dict["local"]["ctx"]}

        self.system_dict["local"]["posenet"] = get_model(model_name, **kwargs)
        self.system_dict["local"]["posenet"].cast('float32')

        self.system_dict["local"]["posenet"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

    def Infer(self, img_path, output_path="result.jpg", bbox_thresh=0.5, kp_thresh=0.2):

        x, img = data.transforms.presets.ssd.load_test(img_path, short=512)
        x = x.copyto(self.system_dict["local"]["ctx"][0]);
        print('Shape of pre-processed image:', x.shape)

        print('Running Person Detector')
        class_IDs, scores, bounding_boxs = self.system_dict["local"]["detector"](x)

        print('Running Pose Estimator')
        pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

        pose_input = pose_input.copyto(self.system_dict["local"]["ctx"][0]);
        predicted_heatmap = self.system_dict["local"]["posenet"](pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        print('Saving Result')
        img = utils.viz.cv_plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(output_path, img)
        print('Done')

        result = {};
        result["pred_coords"] = pred_coords;
        result["confidence"] = confidence;
        result["class_IDs"] = class_IDs;
        result["bounding_boxs"] = bounding_boxs;
        result["scores"] = scores;

        return result;
