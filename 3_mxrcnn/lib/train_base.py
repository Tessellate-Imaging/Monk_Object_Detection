import os
import sys

sys.path.append("mx-rcnn");

import ast
import pprint

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"; 
import mxnet as mx
from mxnet.module import Module

from symdata.loader import AnchorGenerator, AnchorSampler, AnchorLoader
from symnet.logger import logger
from symnet.model import load_param, infer_data_shape, check_shape, initialize_frcnn, get_fixed_params
from symnet.metric import RPNAccMetric, RPNLogLossMetric, RPNL1LossMetric, RCNNAccMetric, RCNNLogLossMetric, RCNNL1LossMetric



system_dict = {};



#######################################################################################################################################
def set_dataset_params(root_dir="data", coco_dir="coco", imageset="traincustom"):
    system_dict["dataset_root"] = root_dir;
    system_dict["dataset"] = coco_dir;
    system_dict["dataset_dir"] = root_dir + "/" + coco_dir;
    system_dict["imageset"] = imageset;
    system_dict["rcnn_num_classes"] = 1;


    

def set_img_preproc_params(img_short_side=600, img_long_side=1000, mean=(123.68, 116.779, 103.939), std=(1.0, 1.0, 1.0)):
    system_dict["img_short_side"] = img_short_side;
    system_dict["img_long_side"] = img_long_side;
    system_dict["img_pixel_means"] = str(mean);
    system_dict["img_pixel_stds"] = str(std);
    system_dict["img_pixel_means"] = ast.literal_eval(system_dict["img_pixel_means"])
    system_dict["img_pixel_stds"] = ast.literal_eval(system_dict["img_pixel_stds"])



#network options - vgg16, renet50, resnet101
def set_model_params(model_name="vgg16", resume=False, start_epoch=0):
    system_dict["network"] = model_name;
    if(model_name == "vgg16"):
        system_dict["pretrained"] = "pretrained/vgg16-0000.params";
        if(not os.path.isdir("pretrained")):  
            os.mkdir("pretrained");
        system_dict["resume"] = resume;
        if(not system_dict["resume"] ):
            if(not os.path.isfile( system_dict["pretrained"])):
                cmd1 = "cp " + os.path.dirname(os.path.realpath(__file__)) + "/download_vgg.sh " + os.getcwd() + "/.";
                os.system(cmd1);
                os.system("chmod +x download_vgg.sh");
                os.system("./download_vgg.sh");
            system_dict["start_epoch"] = 0;
        else:
            system_dict["start_epoch"] = start_epoch;
        system_dict["net_fixed_params"] = '["conv1", "conv2", "conv3", "conv4"]';
        system_dict["rcnn_feat_stride"] = 16;
        system_dict["rcnn_pooled_size"] = '(7, 7)';
    
    elif(model_name == "resnet50"):
        system_dict["pretrained"] = "pretrained/resnet-50-0000.params";
        if(not os.path.isdir("pretrained")):
            os.mkdir("pretrained");
        system_dict["resume"] = resume;
        if(not system_dict["resume"]):
            if(not os.path.isfile( system_dict["pretrained"])):
                cmd1 = "cp " + os.path.dirname(os.path.realpath(__file__)) + "/download_resnet50.sh " + os.getcwd() + "/.";
                os.system(cmd1);
                os.system("chmod +x download_resnet50.sh");
                os.system("./download_resnet50.sh");
            system_dict["start_epoch"] = 0;
        else:
            system_dict["start_epoch"] = start_epoch;
        system_dict["net_fixed_params"] = '["conv0", "stage1", "gamma", "beta"]';
        system_dict["rcnn_feat_stride"] = 16;
        system_dict["rcnn_pooled_size"] = '(14, 14)';

    elif(model_name == "resnet101"):
        system_dict["pretrained"] = "pretrained/resnet-101-0000.params";
        if(not os.path.isdir("pretrained")):
            os.mkdir("pretrained");
        system_dict["resume"] = resume;
        if(not system_dict["resume"]):
            if(not os.path.isfile( system_dict["pretrained"])):
                cmd1 = "cp " + os.path.dirname(os.path.realpath(__file__)) + "/download_resnet101.sh " + os.getcwd() + "/.";
                os.system(cmd1);
                os.system("chmod +x download_resnet101.sh");
                os.system("./download_resnet101.sh");
            system_dict["start_epoch"] = 0;
        else:
            system_dict["start_epoch"] = start_epoch;
        system_dict["net_fixed_params"] = '["conv0", "stage1", "gamma", "beta"]';
        system_dict["rcnn_feat_stride"] = 16;
        system_dict["rcnn_pooled_size"] = '(14, 14)';

    system_dict["net_fixed_params"] = ast.literal_eval(system_dict["net_fixed_params"])



def set_hyper_params(gpus="0", lr=0.001, lr_decay_epoch="7", epochs=10, batch_size=1):
    system_dict["gpus"] = gpus;
    system_dict["lr"] = lr;
    system_dict["lr_decay_epoch"] = lr_decay_epoch;
    system_dict["epochs"] = epochs;
    system_dict["rcnn_batch_size"] = batch_size;


def set_output_params(log_interval=100, save_prefix="model_vgg16"):
    system_dict["log_interval"] = log_interval;
    if(not os.path.isdir("trained_model")):
        os.mkdir("trained_model");
    system_dict["save_prefix"] = "trained_model/" + save_prefix;
#######################################################################################################################################









#######################################################################################################################################
def initialize_rpn_params():
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
    system_dict["rcnn_batch_rois"] = 128;
    system_dict["rcnn_fg_fraction"] = 0.25;
    system_dict["rcnn_fg_overlap"] = 0.5;
    system_dict["rcnn_bbox_stds"] = '(0.1, 0.1, 0.2, 0.2)';

    system_dict["rcnn_pooled_size"] = ast.literal_eval(system_dict["rcnn_pooled_size"])
    system_dict["rcnn_bbox_stds"] = ast.literal_eval(system_dict["rcnn_bbox_stds"])
#######################################################################################################################################











#######################################################################################################################################
def get_coco(system_dict):
    from symimdb.coco import coco
    if not system_dict["imageset"]:
        system_dict["imageset"] = 'train2017'

    isets = system_dict["imageset"].split('+')
    roidb = []
    for iset in isets:
        imdb = coco(iset, system_dict["dataset_root"], system_dict["dataset_dir"])
        system_dict["rcnn_num_classes"] = len(imdb.classes)
        imdb.filter_roidb()
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb



def set_dataset():
    dataset = system_dict["dataset"];
    datasets = {
        dataset: get_coco
    }
    if dataset not in datasets:
        raise ValueError("dataset {} not supported".format(dataset))
    return datasets[dataset](system_dict)
#######################################################################################################################################











#######################################################################################################################################
def get_resnet101_train(system_dict):
    from symnet.symbol_resnet import get_resnet_train

    return get_resnet_train(anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"],
                            rpn_feature_stride=system_dict["rpn_feat_stride"], rpn_pre_topk=system_dict["rpn_pre_nms_topk"],
                            rpn_post_topk=system_dict["rpn_post_nms_topk"], rpn_nms_thresh=system_dict["rpn_nms_thresh"],
                            rpn_min_size=system_dict["rpn_min_size"], rpn_batch_rois=system_dict["rpn_batch_rois"],
                            num_classes=system_dict["rcnn_num_classes"], rcnn_feature_stride=system_dict["rcnn_feat_stride"],
                            rcnn_pooled_size=system_dict["rcnn_pooled_size"], rcnn_batch_size=system_dict["rcnn_batch_size"],
                            rcnn_batch_rois=system_dict["rcnn_batch_rois"], rcnn_fg_fraction=system_dict["rcnn_fg_fraction"],
                            rcnn_fg_overlap=system_dict["rcnn_fg_overlap"], rcnn_bbox_stds=system_dict["rcnn_bbox_stds"],
                            units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))


def get_resnet50_train(system_dict):
    from symnet.symbol_resnet import get_resnet_train

    return get_resnet_train(anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"],
                            rpn_feature_stride=system_dict["rpn_feat_stride"], rpn_pre_topk=system_dict["rpn_pre_nms_topk"],
                            rpn_post_topk=system_dict["rpn_post_nms_topk"], rpn_nms_thresh=system_dict["rpn_nms_thresh"],
                            rpn_min_size=system_dict["rpn_min_size"], rpn_batch_rois=system_dict["rpn_batch_rois"],
                            num_classes=system_dict["rcnn_num_classes"], rcnn_feature_stride=system_dict["rcnn_feat_stride"],
                            rcnn_pooled_size=system_dict["rcnn_pooled_size"], rcnn_batch_size=system_dict["rcnn_batch_size"],
                            rcnn_batch_rois=system_dict["rcnn_batch_rois"], rcnn_fg_fraction=system_dict["rcnn_fg_fraction"],
                            rcnn_fg_overlap=system_dict["rcnn_fg_overlap"], rcnn_bbox_stds=system_dict["rcnn_bbox_stds"],
                            units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


def get_vgg16_train(system_dict):
    from symnet.symbol_vgg import get_vgg_train

    return get_vgg_train(anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"],
                         rpn_feature_stride=system_dict["rpn_feat_stride"], rpn_pre_topk=system_dict["rpn_pre_nms_topk"],
                         rpn_post_topk=system_dict["rpn_post_nms_topk"], rpn_nms_thresh=system_dict["rpn_nms_thresh"],
                         rpn_min_size=system_dict["rpn_min_size"], rpn_batch_rois=system_dict["rpn_batch_rois"],
                         num_classes=system_dict["rcnn_num_classes"], rcnn_feature_stride=system_dict["rcnn_feat_stride"],
                         rcnn_pooled_size=system_dict["rcnn_pooled_size"], rcnn_batch_size=system_dict["rcnn_batch_size"],
                         rcnn_batch_rois=system_dict["rcnn_batch_rois"], rcnn_fg_fraction=system_dict["rcnn_fg_fraction"],
                         rcnn_fg_overlap=system_dict["rcnn_fg_overlap"], rcnn_bbox_stds=system_dict["rcnn_bbox_stds"])

def set_network():
    network = system_dict["network"]
    networks = {
        'vgg16': get_vgg16_train,
        'resnet50': get_resnet50_train,
        'resnet101': get_resnet101_train
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](system_dict)
#######################################################################################################################################







#######################################################################################################################################
def train(sym, roidb):
    # print config
    #logger.info('called with system_dict\n{}'.format(pprint.pformat(vars(system_dict))))
    #print(system_dict)

    # setup multi-gpu
    if(system_dict["gpus"] == "-1"):
        ctx = [mx.cpu(0)];
    else:
        ctx = [mx.gpu(int(i)) for i in system_dict["gpus"].split(',')]
    batch_size = system_dict["rcnn_batch_size"] * len(ctx)

    # load training data
    feat_sym = sym.get_internals()['rpn_cls_score_output']
    ag = AnchorGenerator(feat_stride=system_dict["rpn_feat_stride"],
                         anchor_scales=system_dict["rpn_anchor_scales"], anchor_ratios=system_dict["rpn_anchor_ratios"])
    asp = AnchorSampler(allowed_border=system_dict["rpn_allowed_border"], batch_rois=system_dict["rpn_batch_rois"],
                        fg_fraction=system_dict["rpn_fg_fraction"], fg_overlap=system_dict["rpn_fg_overlap"],
                        bg_overlap=system_dict["rpn_bg_overlap"])
    train_data = AnchorLoader(roidb, batch_size, system_dict["img_short_side"], system_dict["img_long_side"],
                              system_dict["img_pixel_means"], system_dict["img_pixel_stds"], feat_sym, ag, asp, shuffle=True)

    # produce shape max possible
    _, out_shape, _ = feat_sym.infer_shape(data=(1, 3, system_dict["img_long_side"], system_dict["img_long_side"]))
    feat_height, feat_width = out_shape[0][-2:]
    rpn_num_anchors = len(system_dict["rpn_anchor_scales"]) * len(system_dict["rpn_anchor_ratios"])
    data_names = ['data', 'im_info', 'gt_boxes']
    label_names = ['label', 'bbox_target', 'bbox_weight']
    data_shapes = [('data', (batch_size, 3, system_dict["img_long_side"], system_dict["img_long_side"])),
                   ('im_info', (batch_size, 3)),
                   ('gt_boxes', (batch_size, 100, 5))]
    label_shapes = [('label', (batch_size, 1, rpn_num_anchors * feat_height, feat_width)),
                    ('bbox_target', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width)),
                    ('bbox_weight', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width))]

    # print shapes
    data_shape_dict, out_shape_dict = infer_data_shape(sym, data_shapes + label_shapes)
    logger.info('max input shape\n%s' % pprint.pformat(data_shape_dict))
    logger.info('max output shape\n%s' % pprint.pformat(out_shape_dict))

    # load and initialize params
    if system_dict["resume"]:
        arg_params, aux_params = load_param(system_dict["resume"])
    else:
        arg_params, aux_params = load_param(system_dict["pretrained"])
        arg_params, aux_params = initialize_frcnn(sym, data_shapes, arg_params, aux_params)

    # check parameter shapes
    check_shape(sym, data_shapes + label_shapes, arg_params, aux_params)

    # check fixed params
    fixed_param_names = get_fixed_params(sym, system_dict["net_fixed_params"])
    logger.info('locking params\n%s' % pprint.pformat(fixed_param_names))

    # metric
    rpn_eval_metric = RPNAccMetric()
    rpn_cls_metric = RPNLogLossMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    eval_metric = RCNNAccMetric()
    cls_metric = RCNNLogLossMetric()
    bbox_metric = RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=system_dict["log_interval"], auto_reset=False)
    epoch_end_callback = mx.callback.do_checkpoint(system_dict["save_prefix"])

    # learning schedule
    base_lr = system_dict["lr"]
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in system_dict["lr_decay_epoch"].split(',')]
    lr_epoch_diff = [epoch - system_dict["start_epoch"] for epoch in lr_epoch if epoch > system_dict["start_epoch"]]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod = Module(sym, data_names=data_names, label_names=label_names,
                 logger=logger, context=ctx, work_load_list=None,
                 fixed_param_names=fixed_param_names)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore='device',
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=system_dict["start_epoch"], num_epoch=system_dict["epochs"])
#######################################################################################################################################
