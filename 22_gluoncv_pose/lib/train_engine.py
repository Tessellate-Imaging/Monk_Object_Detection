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


class Detector():
    '''
    Class to train a detector

    self.system_dict["params"]:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["params"] = {};

    def Train_Dataset(self, coco_directory):
        self.system_dict["params"]["data_dir"] = data_dir;
        self.system_dict["params"]["train_anno_file"] = 'person_keypoints_train2017';

    def Dataset_Params(self, batch_size=2, num_workers=2, num_joints=17, input_size="256,192",
                        mean='0.485,0.456,0.406', std='0.229,0.224,0.225'):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
        self.system_dict["params"]["num_joints"] = num_joints;
        self.system_dict["params"]["input_size"] = input_size;
        self.system_dict["params"]["mean"] = mean
        self.system_dict["params"]["std"] = std

    def List_Models(self):
        self.system_dict["params"]["model_list"] = ["simple_pose_resnet18_v1b", "simple_pose_resnet50_v1b",
                                                    "simple_pose_resnet101_v1b", "simple_pose_resnet101_v1d",
                                                    "simple_pose_resnet152_v1b", "simple_pose_resnet152_v1d",
                                                    "mobile_pose_resnet18_v1b", "mobile_pose_resnet50_v1b",
                                                    "mobile_pose_mobilenet1.0", "mobile_pose_mobilenetv2_1.0",
                                                    "mobile_pose_mobilenetv3_large", "mobile_pose_mobilenetv3_small",
                                                    "alpha_pose_resnet101_v1b_coco"
                                                    ]
        
        for i in range(len(self.system_dict["params"]["model_list"])):
            print("{}. Model - {}".format(i+1, self.system_dict["params"]["model_list"][i]));

    def List_Modes(self):
        self.system_dict["params"]["mode_list"] = ["symbolic", "imperative", "hybrid"]
        
        for i in range(len(self.system_dict["params"]["mode_list"])):
            print("{}. Modes - {}".format(i+1, self.system_dict["params"]["mode_list"][i]));

    def Model_Params(self, model_name="simple_pose_resnet18_v1b", mode="symbolic", use_pretrained=True, use_pretrained_base=True, num_gpus=1):
        self.system_dict["params"]["model"] = model_name;
        self.system_dict["params"]["mode"] = mode;
        self.system_dict["params"]["use_pretrained"] = use_pretrained;
        self.system_dict["params"]["use_pretrained_base"] = use_pretrained_base;
        self.system_dict["params"]["num_gpus"] = num_gpus;

    def Hyper_Params(self, lr=0.01, weight_decay=0.0001, lr_decay=0.1):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["wd"] = weight_decay;
        self.system_dict["params"]["lr_mode"] = "step";
        self.system_dict["params"]["lr_decay"] = lr_decay;
        self.system_dict["params"]["lr_decay_period"] = 0;

    def Training_Params(self, num_epochs=100, save_frequency=10, output_dir="output", log_iter_interval=100):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["save_frequency"] = save_frequency;
        self.system_dict["params"]["save_dir"] = output_dir;
        self.system_dict["params"]["log_interval"] = log_iter_interval;


    def Train(self):
        if(self.system_dict["params"]["num_epochs"] > 2):
            self.system_dict["params"]["lr_decay_epoch"] = [self.system_dict["params"]["num_epochs"]//3, 2*self.system_dict["params"]["num_epochs"]//3]
        else:
            self.system_dict["params"]["lr_decay_epoch"] = [1]
        self.system_dict["params"]["warmup_lr"] = 0.0;
        self.system_dict["params"]["warmup_epochs"] = 0;
        self.system_dict["params"]["last_gamma"] = False;
        self.system_dict["params"]["no_wd"] = False;
        self.system_dict["params"]["logging_file"] = "keypoints.log";

        filehandler = logging.FileHandler(self.system_dict["params"]["logging_file"])
        streamhandler = logging.StreamHandler()

        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)

        batch_size = self.system_dict["params"]["batch_size"]
        num_joints = self.system_dict["params"]["num_joints"]

        num_gpus = self.system_dict["params"]["num_gpus"]
        batch_size *= max(1, num_gpus)
        context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
        num_workers = self.system_dict["params"]["num_workers"]

        model_name = self.system_dict["params"]["model"]

        kwargs = {'ctx': context, 'num_joints': num_joints,
                  'pretrained': self.system_dict["params"]["use_pretrained"],
                  'pretrained_base': self.system_dict["params"]["use_pretrained_base"],
                  'pretrained_ctx': context}

        net = get_model(model_name, **kwargs)
        net.cast(self.system_dict["params"]["dtype"])



        input_size = [int(i) for i in self.system_dict["params"]["input_size"].split(',')]
        train_dataset, train_data,  train_batch_fn = self.get_data_loader(self.system_dict["params"]["data_dir"], batch_size,
                                                                     num_workers, input_size)

        num_training_samples = len(train_dataset)


        lr_decay = self.system_dict["params"]["lr_decay"]
        lr_decay_period = self.system_dict["params"]["lr_decay_period"]
        if self.system_dict["params"]["lr_decay_period"] > 0:
            lr_decay_epoch = list(range(lr_decay_period, self.system_dict["params"]["num_epochs"], lr_decay_period))
        else:
            lr_decay_epoch = self.system_dict["params"]["lr_decay_epoch"]
        lr_decay_epoch = [e - self.system_dict["params"]["warmup_epochs"] for e in lr_decay_epoch]
        num_batches = num_training_samples // batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self.system_dict["params"]["lr"],
                        nepochs=self.system_dict["params"]["warmup_epochs"], iters_per_epoch=num_batches),
            LRScheduler(self.system_dict["params"]["lr_mode"], base_lr=self.system_dict["params"]["lr"], target_lr=0,
                        nepochs=self.system_dict["params"]["num_epochs"] - self.system_dict["params"]["warmup_epochs"],
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=lr_decay, power=2)
        ])

        # optimizer = 'sgd'
        # optimizer_params = {'wd': self.system_dict["params"]["wd"], 'momentum': 0.9, 'lr_scheduler': lr_scheduler}
        optimizer = 'adam'
        optimizer_params = {'wd': self.system_dict["params"]["wd"], 'lr_scheduler': lr_scheduler}
        if self.system_dict["params"]["dtype"] != 'float32':
            optimizer_params['multi_precision'] = True

        save_frequency = self.system_dict["params"]["save_frequency"]
        if self.system_dict["params"]["save_dir"] and save_frequency:
            save_dir = self.system_dict["params"]["save_dir"]
            makedirs(save_dir)
        else:
            save_dir = ''
            save_frequency = 0


        net = self.train(context)



    def get_data_loader(self, data_dir, batch_size, num_workers, input_size):

        def train_batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            weight = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            imgid = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
            return data, label, weight, imgid

        train_dataset = mscoco.keypoints.COCOKeyPoints(data_dir, splits=('person_keypoints_train2017'))
        heatmap_size = [int(i/4) for i in input_size]

        meanvec = [float(i) for i in self.system_dict["params"]["mean"].split(',')]
        stdvec = [float(i) for i in self.system_dict["params"]["std"].split(',')]
        transform_train = SimplePoseDefaultTrainTransform(num_joints=train_dataset.num_joints,
                                                          joint_pairs=train_dataset.joint_pairs,
                                                          image_size=input_size, heatmap_size=heatmap_size,
                                                          sigma=self.system_dict["params"]["sigma"], scale_factor=0.30, rotation_factor=40,
                                                          mean=meanvec, std=stdvec, random_flip=True)

        train_data = gluon.data.DataLoader(
            train_dataset.transform(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        return train_dataset, train_data, train_batch_fn


    def train(self, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if self.system_dict["params"]["use_pretrained_base"]:
            if model_name.startswith('simple'):
                net.deconv_layers.initialize(ctx=ctx)
                net.final_layer.initialize(ctx=ctx)
            elif model_name.startswith('mobile'):
                net.upsampling.initialize(ctx=ctx)
        else:
            net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

        L = gluon.loss.L2Loss()
        metric = HeatmapAccuracy()

        best_val_score = 1

        if self.system_dict["params"]["mode"] == 'hybrid':
            net.hybridize(static_alloc=True, static_shape=True)

        for epoch in range(self.system_dict["params"]["num_epochs"]):
            loss_val = 0
            tic = time.time()
            btic = time.time()
            metric.reset()

            for i, batch in enumerate(train_data):
                data, label, weight, imgid = train_batch_fn(batch, ctx)

                with ag.record():
                    outputs = [net(X.astype(self.system_dict["params"]["dtype"], copy=False)) for X in data]
                    loss = [nd.cast(L(nd.cast(yhat, 'float32'), y, w), self.system_dict["params"]["dtype"])
                            for yhat, y, w in zip(outputs, label, weight)]
                ag.backward(loss)
                trainer.step(batch_size)

                metric.update(label, outputs)

                loss_val += sum([l.mean().asscalar() for l in loss]) / num_gpus
                if self.system_dict["params"]["log_interval"] and not (i+1)%self.system_dict["params"]["log_interval"]:
                    metric_name, metric_score = metric.get()
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tloss=%f\tlr=%f\t%s=%.3f'%(
                                 epoch, i, batch_size*self.system_dict["params"]["log_interval"]/(time.time()-btic),
                                 loss_val / (i+1), trainer.learning_rate, metric_name, metric_score))
                    btic = time.time()

            time_elapsed = time.time() - tic
            logger.info('Epoch[%d]\t\tSpeed: %d samples/sec over %d secs\tloss=%f\n'%(
                         epoch, int(i*batch_size / time_elapsed), int(time_elapsed), loss_val / (i+1)))
            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/%s-%d.params'%(save_dir, model_name, epoch))
                trainer.save_states('%s/%s-%d.states'%(save_dir, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/%s-%d.params'%(save_dir, model_name, self.system_dict["params"]["num_epochs"]-1))
            trainer.save_states('%s/%s-%d.states'%(save_dir, model_name, self.system_dict["params"]["num_epochs"]-1))

        return net















