import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz
import pandas as pd
import cv2
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
if(isnotebook()):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm as tqdm


from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform

class Detector():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["model_set_1"] = ["ssd_300_vgg16_atrous_coco", "ssd_300_vgg16_atrous_voc"];
        self.system_dict["model_set_2"] = ["ssd_512_vgg16_atrous_coco", "ssd_512_vgg16_atrous_voc"];
        self.system_dict["model_set_3"] = ["ssd_512_resnet50_v1_coco", "ssd_512_resnet50_v1_voc"];
        self.system_dict["model_set_4"] = ["ssd_512_mobilenet1.0_voc", "ssd_512_mobilenet1.0_coco"];
        self.system_dict["model_set_5"] = ["yolo3_darknet53_voc", "yolo3_darknet53_coco"];
        self.system_dict["model_set_6"] = ["yolo3_mobilenet1.0_voc", "yolo3_mobilenet1.0_coco"];



    def Dataset(self, root, img_dir, anno_file, batch_size=4, num_workers=0):
        self.system_dict["root"] = root;
        self.system_dict["img_dir"] = img_dir;
        self.system_dict["anno_file"] = anno_file;
        self.system_dict["batch_size"] = batch_size;
        self.system_dict["num_workers"] = num_workers;

        df = pd.read_csv(self.system_dict["root"] + "/" + self.system_dict["anno_file"]);
        columns = df.columns;


        classes = [];
        for i in tqdm(range(len(df))):
            tmp = df["Label"][i].split(" ");
            for j in range(len(tmp)//5):
                label = tmp[j*5+4];
                if(label not in classes):
                    classes.append(label)
                    
        self.system_dict["classes"] = sorted(classes)

        with open('train.lst', 'w') as fw:    
            for i in tqdm(range(len(df))):
                img_name = df[columns[0]][i];
                tmp = df[columns[1]][i].split(" ");
                class_names = [];
                bbox = [];
                ids = [];
                for j in range(len(tmp)//5):
                    x1 = int(float(tmp[j*5+0]));
                    y1 = int(float(tmp[j*5+1]));
                    x2 = int(float(tmp[j*5+2]));
                    y2 = int(float(tmp[j*5+3]));
                    label = tmp[j*5+4]
                    class_names.append(label);
                    bbox.append([x1, y1, x2, y2]);
                    ids.append(classes.index(label));

                if(not os.path.isfile(self.system_dict["root"] + "/" + self.system_dict["img_dir"] + "/" + img_name)):
                	continue;
                
                img = cv2.imread(self.system_dict["root"] + "/" + self.system_dict["img_dir"] + "/" + img_name); 
                all_boxes = np.array(bbox);
                all_ids = np.array(ids);
                line = self.write_line(img_name, img.shape, all_boxes, all_ids, i)
                fw.write(line)

        cmd1 = "cp " + os.path.dirname(os.path.realpath(__file__)) + "/im2rec.py " + os.getcwd() + "/";
        os.system(cmd1);  

        cmd2 = "python im2rec.py train.lst " + self.system_dict["root"] + "/" + self.system_dict["img_dir"] + "/ --pass-through --pack-label"
        os.system(cmd2);

        self.system_dict["local"]["train_dataset"] = gcv.data.RecordFileDetection('train.rec')



    def write_line(self, img_path, im_shape, boxes, ids, idx):
        h, w, c = im_shape
        # for header, we use minimal length 2, plus width and height
        # with A: 4, B: 5, C: width, D: height
        A = 4
        B = 5
        C = w
        D = h
        # concat id and bboxes
        labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
        # normalized bboxes (recommanded)
        labels[:, (1, 3)] /= float(w)
        labels[:, (2, 4)] /= float(h)
        # flatten
        labels = labels.flatten().tolist()
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = [str(x) for x in labels]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line


    def Model(self, model_name, use_pretrained=True, use_gpu=True, gpu_devices=[0]):
        self.system_dict["model_name"] = model_name;
        self.system_dict["use_pretrained"] = use_pretrained;
        if(self.system_dict["model_name"] in self.system_dict["model_set_1"]):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model(self.system_dict["model_name"], 
                pretrained=self.system_dict["use_pretrained"]);
            self.system_dict["local"]["net"].reset_class(self.system_dict["classes"])
            self.system_dict["img_shape"] = (300, 300); 

            width, height = self.system_dict["img_shape"][0], self.system_dict["img_shape"][1]
            with autograd.train_mode():
                _, _, anchors = self.system_dict["local"]["net"](mx.nd.zeros((1, 3, height, width)))

            batchify_fn = Tuple(Stack(), Stack(), Stack())
            self.system_dict["local"]["train_loader"] = gluon.data.DataLoader(
                self.system_dict["local"]["train_dataset"].transform(SSDDefaultTrainTransform(width, height, anchors)),
                self.system_dict["batch_size"], True, batchify_fn=batchify_fn, last_batch='rollover', 
                num_workers=self.system_dict["num_workers"])

            self.set_device(use_gpu=use_gpu ,gpu_devices=gpu_devices);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif((self.system_dict["model_name"] in self.system_dict["model_set_2"]) or (self.system_dict["model_name"] in self.system_dict["model_set_3"])
            or (self.system_dict["model_name"] in self.system_dict["model_set_4"])):
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model(self.system_dict["model_name"], 
                pretrained=self.system_dict["use_pretrained"]);
            self.system_dict["local"]["net"].reset_class(self.system_dict["classes"])
            self.system_dict["img_shape"] = (512, 512); 

            width, height = self.system_dict["img_shape"][0], self.system_dict["img_shape"][1]
            with autograd.train_mode():
                _, _, anchors = self.system_dict["local"]["net"](mx.nd.zeros((1, 3, height, width)))

            batchify_fn = Tuple(Stack(), Stack(), Stack())
            self.system_dict["local"]["train_loader"] = gluon.data.DataLoader(
                self.system_dict["local"]["train_dataset"].transform(SSDDefaultTrainTransform(width, height, anchors)),
                self.system_dict["batch_size"], True, batchify_fn=batchify_fn, last_batch='rollover', 
                num_workers=self.system_dict["num_workers"])

            self.set_device(use_gpu=use_gpu, gpu_devices=gpu_devices);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])

        elif((self.system_dict["model_name"] in self.system_dict["model_set_5"]) or (self.system_dict["model_name"] in self.system_dict["model_set_6"])) :
            self.system_dict["local"]["net"] = gcv.model_zoo.get_model(self.system_dict["model_name"], 
                pretrained=self.system_dict["use_pretrained"]);
            self.system_dict["local"]["net"].reset_class(self.system_dict["classes"])
            self.system_dict["img_shape"] = (416, 416); 

            width, height = self.system_dict["img_shape"][0], self.system_dict["img_shape"][1]

            train_transform = YOLO3DefaultTrainTransform(width, height, self.system_dict["local"]["net"])
            batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))

            self.system_dict["local"]["train_loader"] = gluon.data.DataLoader(
                self.system_dict["local"]["train_dataset"].transform(train_transform),
                self.system_dict["batch_size"], True, batchify_fn=batchify_fn, last_batch='rollover', 
                num_workers=self.system_dict["num_workers"])

            self.set_device(use_gpu=use_gpu, gpu_devices=gpu_devices);
            self.system_dict["local"]["net"].collect_params().reset_ctx(self.system_dict["local"]["ctx"])





    def set_device(self, use_gpu=True ,gpu_devices=[0]):
        self.system_dict["use_gpu"] = use_gpu;
        
        if self.system_dict["use_gpu"]:
            self.system_dict["gpu_devices"] = gpu_devices;
            self.system_dict["local"]["ctx"]= [mx.gpu(int(i)) for i in self.system_dict["gpu_devices"]]
        else:
            self.system_dict["local"]["ctx"] = [mx.cpu()]


    def Set_Learning_Rate(self, lr):
        self.system_dict["local"]["trainer"] = gluon.Trainer(
            self.system_dict["local"]["net"].collect_params(), 'sgd',
            {'learning_rate': lr, 'wd': 0.0005, 'momentum': 0.9})

        if((self.system_dict["model_name"] in self.system_dict["model_set_1"]) or (self.system_dict["model_name"] in self.system_dict["model_set_2"])
            or (self.system_dict["model_name"] in self.system_dict["model_set_3"]) or (self.system_dict["model_name"] in self.system_dict["model_set_4"])):
            self.system_dict["local"]["mbox_loss"] = gcv.loss.SSDMultiBoxLoss()
            self.system_dict["local"]["ce_metric"] = mx.metric.Loss('CrossEntropy')
            self.system_dict["local"]["smoothl1_metric"] = mx.metric.Loss('SmoothL1');

        elif((self.system_dict["model_name"] in self.system_dict["model_set_5"]) or (self.system_dict["model_name"] in self.system_dict["model_set_6"])):
            self.system_dict["local"]["loss"] = gcv.loss.YOLOV3Loss()
            self.system_dict["local"]["obj_metrics"] = mx.metric.Loss('ObjLoss')
            self.system_dict["local"]["center_metrics"] = mx.metric.Loss('BoxCenterLoss')
            self.system_dict["local"]["scale_metrics"] = mx.metric.Loss('BoxScaleLoss')
            self.system_dict["local"]["cls_metrics"] = mx.metric.Loss('ClassLoss')


    def Train(self, epochs, params_file):
        self.system_dict["num_epochs"] = epochs;
        self.system_dict["params_file"] = params_file;
        self.system_dict["training_metrics"] = [];
        self.system_dict["training_time"] = 0.0;

        if((self.system_dict["model_name"] in self.system_dict["model_set_1"]) or (self.system_dict["model_name"] in self.system_dict["model_set_2"])
            or (self.system_dict["model_name"] in self.system_dict["model_set_3"]) or (self.system_dict["model_name"] in self.system_dict["model_set_4"])):
            for epoch in range(self.system_dict["num_epochs"]):
                self.system_dict["local"]["ce_metric"].reset()
                self.system_dict["local"]["smoothl1_metric"].reset()
                tic = time.time()
                btic = time.time()
                self.system_dict["local"]["net"].hybridize(static_alloc=True, static_shape=True)
                for i, batch in enumerate(self.system_dict["local"]["train_loader"]):
                    batch_size = batch[0].shape[0]
                    data = gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], 
                        batch_axis=0)
                    cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], 
                        batch_axis=0)
                    box_targets = gluon.utils.split_and_load(batch[2], ctx_list=self.system_dict["local"]["ctx"], 
                        batch_axis=0)
                    with autograd.record():
                        cls_preds = []
                        box_preds = []
                        for x in data:
                            cls_pred, box_pred, _ = self.system_dict["local"]["net"](x)
                            cls_preds.append(cls_pred)
                            box_preds.append(box_pred)
                        sum_loss, cls_loss, box_loss = self.system_dict["local"]["mbox_loss"](
                            cls_preds, box_preds, cls_targets, box_targets)
                        autograd.backward(sum_loss)
                    # since we have already normalized the loss, we don't want to normalize
                    # by batch-size anymore
                    self.system_dict["local"]["trainer"].step(batch_size)
                    self.system_dict["local"]["ce_metric"].update(0, [l * batch_size for l in cls_loss])
                    self.system_dict["local"]["smoothl1_metric"].update(0, [l * batch_size for l in box_loss])
                    name1, loss1 = self.system_dict["local"]["ce_metric"].get()
                    name2, loss2 = self.system_dict["local"]["smoothl1_metric"].get()
                    if i % 20 == 0:
                        print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
                        tmp = {};
                        tmp["epoch"] = epoch;
                        tmp["batch"] = batch;
                        tmp["name1"] = loss1;
                        tmp["name2"] = loss2;
                        self.system_dict["training_metrics"].append(tmp);

                    btic = time.time()
                self.system_dict["training_time"] += time.time() - tic;

            self.system_dict["local"]["net"].save_parameters(self.system_dict["params_file"])

        if((self.system_dict["model_name"] in self.system_dict["model_set_5"]) or (self.system_dict["model_name"] in self.system_dict["model_set_6"])):
            for epoch in range(self.system_dict["num_epochs"]):
                tic = time.time()
                btic = time.time()
                self.system_dict["local"]["net"].hybridize(static_alloc=True, static_shape=True)
                self.system_dict["local"]["obj_metrics"].reset()
                self.system_dict["local"]["center_metrics"].reset()
                self.system_dict["local"]["scale_metrics"].reset()
                self.system_dict["local"]["cls_metrics"].reset()
                
                
                for i, batch in enumerate(self.system_dict["local"]["train_loader"]):
                    batch_size = batch[0].shape[0]
                    data = gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], 
                        batch_axis=0)
                    # objectness, center_targets, scale_targets, weights, class_targets
                    fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=self.system_dict["local"]["ctx"], 
                        batch_axis=0) for it in range(1, 6)]
                    gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=self.system_dict["local"]["ctx"], 
                        batch_axis=0)
                    sum_losses = []
                    obj_losses = []
                    center_losses = []
                    scale_losses = []
                    cls_losses = []
                    with autograd.record():
                        for ix, x in enumerate(data):
                            obj_loss, center_loss, scale_loss, cls_loss = self.system_dict["local"]["net"](x, gt_boxes[ix], 
                                *[ft[ix] for ft in fixed_targets])
                            sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                            obj_losses.append(obj_loss)
                            center_losses.append(center_loss)
                            scale_losses.append(scale_loss)
                            cls_losses.append(cls_loss)
                        autograd.backward(sum_losses)
                    self.system_dict["local"]["trainer"].step(batch_size)
                    self.system_dict["local"]["obj_metrics"].update(0, obj_losses)
                    self.system_dict["local"]["center_metrics"].update(0, center_losses)
                    self.system_dict["local"]["scale_metrics"].update(0, scale_losses)
                    self.system_dict["local"]["cls_metrics"].update(0, cls_losses)

                    if i % 20 == 0:
                        name1, loss1 = self.system_dict["local"]["obj_metrics"].get()
                        name2, loss2 = self.system_dict["local"]["center_metrics"].get()
                        name3, loss3 = self.system_dict["local"]["scale_metrics"].get()
                        name4, loss4 = self.system_dict["local"]["cls_metrics"].get()
                        tmp = {};
                        tmp["epoch"] = epoch;
                        tmp["batch"] = batch;
                        tmp["name1"] = loss1;
                        tmp["name2"] = loss2;
                        tmp["name3"] = loss3;
                        tmp["name4"] = loss4;
                        self.system_dict["training_metrics"].append(tmp);


                        print('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, self.system_dict["local"]["trainer"].learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, 
                    loss3, name4, loss4))
                    btic = time.time()
                self.system_dict["training_time"] += time.time() - tic;

            self.system_dict["local"]["net"].save_parameters(self.system_dict["params_file"])
