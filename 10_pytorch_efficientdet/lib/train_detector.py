import datetime
import os
import argparse
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss




class Detector():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;
        self.system_dict["params"] = {};

        self.set_fixed_params();

    
    def set_fixed_params(self):
        self.system_dict["params"]["project"] = "custom";
        self.system_dict["params"]["project_name"] = "custom";
        self.system_dict["params"]["mean"] = [0.485, 0.456, 0.406];
        self.system_dict["params"]["std"] = [0.229, 0.224, 0.225];
        self.system_dict["params"]["anchors_scales"] = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]';
        self.system_dict["params"]["anchors_ratios"] = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]';
        self.system_dict["params"]["log_path"] = "logs/";
        self.system_dict["params"]["saved_path"] = "trained_weights/";
        self.system_dict["params"]["debug"] = False;


    def set_train_dataset(self, root_dir, coco_dir, img_dir, set_dir,  classes_list=[], batch_size=2, num_workers=3):
        self.system_dict["dataset"]["train"]["root_dir"] = root_dir
        self.system_dict["dataset"]["train"]["coco_dir"] = coco_dir
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir
        self.system_dict["dataset"]["train"]["set_dir"] = set_dir

        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
        self.system_dict["params"]["obj_list"] = classes_list;


    def set_val_dataset(self, root_dir, coco_dir, img_dir, set_dir):
        self.system_dict["dataset"]["val"]["root_dir"] = root_dir
        self.system_dict["dataset"]["val"]["coco_dir"] = coco_dir
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir
        self.system_dict["dataset"]["val"]["set_dir"] = set_dir

        self.system_dict["dataset"]["val"]["status"] = True;


    #"efficientdet-d0.pth";
    #"efficientdet-d1.pth";
    #"efficientdet-d2.pth";
    #"efficientdet-d3.pth";
    #"efficientdet-d4.pth";
    #"efficientdet-d5.pth";
    #"efficientdet-d6.pth";
    #"efficientdet-d7.pth";
    def set_model(self, model_name="efficientdet-d0.pth", num_gpus=1, freeze_head=False):
        if("0" in model_name):
            self.system_dict["params"]["compound_coef"] = 0;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d0.pth";
        elif("1" in model_name):
            self.system_dict["params"]["compound_coef"] = 1;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d1.pth";
        elif("2" in model_name):
            self.system_dict["params"]["compound_coef"] = 2;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d2.pth";
        elif("3" in model_name):
            self.system_dict["params"]["compound_coef"] = 3;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d3.pth";
        elif("4" in model_name):
            self.system_dict["params"]["compound_coef"] = 4;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d4.pth";
        elif("5" in model_name):
            self.system_dict["params"]["compound_coef"] = 5;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d5.pth";
        elif("6" in model_name):
            self.system_dict["params"]["compound_coef"] = 6;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d6.pth";
        elif("7" in model_name):
            self.system_dict["params"]["compound_coef"] = 7;
            self.system_dict["params"]["load_weights"] = "pretrained_weights/efficientdet-d7.pth";

        self.system_dict["params"]["num_gpus"] = num_gpus;
        self.system_dict["params"]["head_only"] = freeze_head;
        

    #adamw
    #sgd
    def set_hyperparams(self, optimizer="adamw", lr=0.001, es_min_delta=0.0, es_patience=0):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["optim"] = optimizer; 
        self.system_dict["params"]["es_min_delta"] = es_min_delta;
        self.system_dict["params"]["es_patience"] = es_patience;


    def train(self, num_epochs=2, val_interval=1, save_interval=1):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["val_interval"] = val_interval;
        self.system_dict["params"]["save_interval"] = save_interval;

        self.start_training();


    def start_training(self):
        if self.system_dict["params"]["num_gpus"] == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        else:
            torch.manual_seed(42)

        self.system_dict["params"]["saved_path"] = self.system_dict["params"]["saved_path"] + "/" + self.system_dict["params"]["project_name"] + "/";
        self.system_dict["params"]["log_path"] = self.system_dict["params"]["log_path"] + "/" + self.system_dict["params"]["project_name"] + "/tensorboard/";
        os.makedirs(self.system_dict["params"]["saved_path"], exist_ok=True)
        os.makedirs(self.system_dict["params"]["log_path"], exist_ok=True)

        training_params = {'batch_size': self.system_dict["params"]["batch_size"],
                   'shuffle': True,
                   'drop_last': True,
                   'collate_fn': collater,
                   'num_workers': self.system_dict["params"]["num_workers"]}

        val_params = {'batch_size': self.system_dict["params"]["batch_size"],
                      'shuffle': False,
                      'drop_last': True,
                      'collate_fn': collater,
                      'num_workers': self.system_dict["params"]["num_workers"]}

        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        training_set = CocoDataset(self.system_dict["dataset"]["train"]["root_dir"], 
                                    self.system_dict["dataset"]["train"]["coco_dir"], 
                                    self.system_dict["dataset"]["train"]["img_dir"], 
                                    set_dir=self.system_dict["dataset"]["train"]["set_dir"],
                                    transform=transforms.Compose([Normalizer(mean=self.system_dict["params"]["mean"], std=self.system_dict["params"]["std"]),
                                                                 Augmenter(),
                                                                 Resizer(input_sizes[self.system_dict["params"]["compound_coef"]])]))
        training_generator = DataLoader(training_set, **training_params)

        if(self.system_dict["dataset"]["val"]["status"]):
            val_set = CocoDataset(self.system_dict["dataset"]["val"]["root_dir"], 
                                    self.system_dict["dataset"]["val"]["coco_dir"], 
                                    self.system_dict["dataset"]["val"]["img_dir"], 
                                    set_dir=self.system_dict["dataset"]["val"]["set_dir"],
                                    transform=transforms.Compose([Normalizer(self.system_dict["params"]["mean"], self.system_dict["params"]["std"]),
                                                             Resizer(input_sizes[self.system_dict["params"]["compound_coef"]])]))
            val_generator = DataLoader(val_set, **val_params)

        print("");
        print("");
        model = EfficientDetBackbone(num_classes=len(self.system_dict["params"]["obj_list"]), 
                                        compound_coef=self.system_dict["params"]["compound_coef"],
                                        ratios=eval(self.system_dict["params"]["anchors_ratios"]), 
                                        scales=eval(self.system_dict["params"]["anchors_scales"]));

        os.makedirs("pretrained_weights", exist_ok=True);

        if(self.system_dict["params"]["compound_coef"] == 0):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth -O " + \
                            self.system_dict["params"]["load_weights"];
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 1):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 2):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 3):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 4):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 5):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 6):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);
        elif(self.system_dict["params"]["compound_coef"] == 7):
            if(not os.path.isfile(self.system_dict["params"]["load_weights"])):
                print("Downloading weights");
                cmd = "wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth -O " + \
                            self.system_dict["params"]["load_weights"]
                os.system(cmd);



        

        # load last weights
        if self.system_dict["params"]["load_weights"] is not None:
            if self.system_dict["params"]["load_weights"].endswith('.pth'):
                weights_path = self.system_dict["params"]["load_weights"]
            else:
                weights_path = get_last_weights(self.system_dict["params"]["saved_path"])
            try:
                last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
            except:
                last_step = 0


            try:
                ret = model.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
                print(
                    '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

            print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
        else:
            last_step = 0
            print('[Info] initializing weights...')
            init_weights(model)

        print("");
        print("");

        # freeze backbone if train head_only
        if self.system_dict["params"]["head_only"]:
            def freeze_backbone(m):
                classname = m.__class__.__name__
                for ntl in ['EfficientNet', 'BiFPN']:
                    if ntl in classname:
                        for param in m.parameters():
                            param.requires_grad = False

            model.apply(freeze_backbone)
            print('[Info] freezed backbone')

        print("");
        print("");

        if self.system_dict["params"]["num_gpus"] > 1 and self.system_dict["params"]["batch_size"] // self.system_dict["params"]["num_gpus"] < 4:
            model.apply(replace_w_sync_bn)
            use_sync_bn = True
        else:
            use_sync_bn = False

        writer = SummaryWriter(self.system_dict["params"]["log_path"] + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

        model = ModelWithLoss(model, debug=self.system_dict["params"]["debug"])

        if self.system_dict["params"]["num_gpus"] > 0:
            model = model.cuda()
            if self.system_dict["params"]["num_gpus"] > 1:
                model = CustomDataParallel(model, self.system_dict["params"]["num_gpus"])
                if use_sync_bn:
                    patch_replication_callback(model)

        if self.system_dict["params"]["optim"] == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), self.system_dict["params"]["lr"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), self.system_dict["params"]["lr"], momentum=0.9, nesterov=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        epoch = 0
        best_loss = 1e5
        best_epoch = 0
        step = max(0, last_step)
        model.train()

        num_iter_per_epoch = len(training_generator)


        try:
            for epoch in range(self.system_dict["params"]["num_epochs"]):
                last_epoch = step // num_iter_per_epoch
                if epoch < last_epoch:
                    continue

                epoch_loss = []
                progress_bar = tqdm(training_generator)
                for iter, data in enumerate(progress_bar):
                    if iter < step - last_epoch * num_iter_per_epoch:
                        progress_bar.update()
                        continue
                    try:
                        imgs = data['img']
                        annot = data['annot']

                        if self.system_dict["params"]["num_gpus"] == 1:
                            # if only one gpu, just send it to cuda:0
                            # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        optimizer.zero_grad()
                        cls_loss, reg_loss = model(imgs, annot, obj_list=self.system_dict["params"]["obj_list"])
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        optimizer.step()

                        epoch_loss.append(float(loss))

                        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                                step, epoch, self.system_dict["params"]["num_epochs"], iter + 1, num_iter_per_epoch, cls_loss.item(),
                                reg_loss.item(), loss.item()))
                        writer.add_scalars('Loss', {'train': loss}, step)
                        writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                        writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                        # log learning_rate
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('learning_rate', current_lr, step)

                        step += 1

                        if step % self.system_dict["params"]["save_interval"] == 0 and step > 0:
                            self.save_checkpoint(model, f'efficientdet-d{self.system_dict["params"]["compound_coef"]}_trained.pth')
                            #print('checkpoint...')

                    except Exception as e:
                        print('[Error]', traceback.format_exc())
                        print(e)
                        continue
                scheduler.step(np.mean(epoch_loss))

                if epoch % self.system_dict["params"]["val_interval"] == 0 and self.system_dict["dataset"]["val"]["status"]:
                    print("Running validation");
                    model.eval()
                    loss_regression_ls = []
                    loss_classification_ls = []
                    for iter, data in enumerate(val_generator):
                        with torch.no_grad():
                            imgs = data['img']
                            annot = data['annot']

                            if self.system_dict["params"]["num_gpus"] == 1:
                                imgs = imgs.cuda()
                                annot = annot.cuda()

                            cls_loss, reg_loss = model(imgs, annot, obj_list=self.system_dict["params"]["obj_list"])
                            cls_loss = cls_loss.mean()
                            reg_loss = reg_loss.mean()

                            loss = cls_loss + reg_loss
                            if loss == 0 or not torch.isfinite(loss):
                                continue

                            loss_classification_ls.append(cls_loss.item())
                            loss_regression_ls.append(reg_loss.item())

                    cls_loss = np.mean(loss_classification_ls)
                    reg_loss = np.mean(loss_regression_ls)
                    loss = cls_loss + reg_loss

                    print(
                        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                            epoch, self.system_dict["params"]["num_epochs"], cls_loss, reg_loss, loss))
                    writer.add_scalars('Loss', {'val': loss}, step)
                    writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                    if loss + self.system_dict["params"]["es_min_delta"] < best_loss:
                        best_loss = loss
                        best_epoch = epoch

                        self.save_checkpoint(model, f'efficientdet-d{self.system_dict["params"]["compound_coef"]}_trained.pth')

                    model.train()

                    # Early stopping
                    if epoch - best_epoch > self.system_dict["params"]["es_patience"] > 0:
                        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                        break
        except KeyboardInterrupt:
            self.save_checkpoint(model, f'efficientdet-d{self.system_dict["params"]["compound_coef"]}_trained.pth')
            writer.close()
        writer.close()


        print("");
        print("");
        print("Training complete");





    def save_checkpoint(self, model, name):
        if isinstance(model, CustomDataParallel):
            torch.save(model.module.model.state_dict(), os.path.join(self.system_dict["params"]["saved_path"], name))
        else:
            torch.save(model.model.state_dict(), os.path.join(self.system_dict["params"]["saved_path"], name))