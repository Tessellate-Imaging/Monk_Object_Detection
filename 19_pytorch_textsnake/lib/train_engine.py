import os
import time
from datetime import datetime
import scipy.io as io
import numpy as np
from tqdm import tqdm

import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from util.shedule import FixLR

from dataset.total_text import TotalText, TotalText_txt, TotalText_mat
from dataset.synth_text import SynthText
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import BaseTransform, Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary



class Detector():
    '''
    Class to train a detector
    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["params"] = {};
        self.system_dict["params"]["val_dataset"] = False;
        
        
    def save_model(self, model, epoch, lr, optimzer):
        save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
        if not os.path.exists(save_dir):
            mkdirs(save_dir)

        save_path = os.path.join(save_dir, 'textsnake_{}_{}.pth'.format(model.backbone_name, epoch))
        print('Saving to {}.'.format(save_path))
        state_dict = {
            'lr': lr,
            'epoch': epoch,
            'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
            'optimizer': optimzer.state_dict()
        }
        torch.save(state_dict, save_path)


    def load_model(self, model, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['model'])
        
        
    def train(self, model, train_loader, criterion, scheduler, optimizer, epoch, logger, train_step):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        model.train()
        scheduler.step()
        lr = scheduler.get_lr()[0];
        print('Epoch: {} : LR = {}'.format(epoch, lr))

        for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(train_loader):
            data_time.update(time.time() - end)

            train_step += 1

            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

            output = model(img)
            tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
                criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
            loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if cfg.viz and i % cfg.viz_freq == 0:
                visualize_network_output(output, tr_mask, tcl_mask, mode='train')

            if i % cfg.display_freq == 0:
                #print(loss.item())
                #print(tr_loss.item())
                #print(tcl_loss.item())
                #print(sin_loss.item())
                #print(cos_loss.item())
                #print(radii_loss.item())
                try:
                    print('({:d} / {:d}) - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                        i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(), cos_loss.item(), radii_loss.item())
                    )
                except:
                    print('({:d} / {:d}) - Loss: {:.4f} - tr_loss: {:.4f}'.format(
                        i, len(train_loader), loss.item(), tr_loss.item())
                    )

            if i % cfg.log_freq == 0:
                try:
                    logger.write_scalars({
                        'loss': loss.item(),
                        'tr_loss': tr_loss.item(),
                        'tcl_loss': tcl_loss.item(),
                        'sin_loss': sin_loss.item(),
                        'cos_loss': cos_loss.item(),
                        'radii_loss': radii_loss.item()
                    }, tag='train', n_iter=train_step)
                except:
                    logger.write_scalars({
                        'loss': loss.item(),
                        'tr_loss': tr_loss.item()
                    }, tag='train', n_iter=train_step)

        if epoch % cfg.save_freq == 0:
            self.save_model(model, epoch, scheduler.get_lr(), optimizer)

        print('Training Loss: {}'.format(losses.avg))

        return train_step
    
    
    
    def validation(self, model, valid_loader, criterion, epoch, logger):
        with torch.no_grad():
            model.eval()
            losses = AverageMeter()
            tr_losses = AverageMeter()
            tcl_losses = AverageMeter()
            sin_losses = AverageMeter()
            cos_losses = AverageMeter()
            radii_losses = AverageMeter()

            for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(valid_loader):

                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
                    img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

                output = model(img)

                tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
                    criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
                loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

                # update losses
                losses.update(loss.item())
                tr_losses.update(tr_loss.item())
                tcl_losses.update(tcl_loss.item())
                sin_losses.update(sin_loss.item())
                cos_losses.update(cos_loss.item())
                radii_losses.update(radii_loss.item())

                if cfg.viz and i % cfg.viz_freq == 0:
                    visualize_network_output(output, tr_mask, tcl_mask, mode='val')

                if i % cfg.display_freq == 0:
                    print(
                        'Validation: - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                            loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(),
                            cos_loss.item(), radii_loss.item())
                    )

            logger.write_scalars({
                'loss': losses.avg,
                'tr_loss': tr_losses.avg,
                'tcl_loss': tcl_losses.avg,
                'sin_loss': sin_losses.avg,
                'cos_loss': cos_losses.avg,
                'radii_loss': radii_losses.avg
            }, tag='val', n_iter=epoch)

            print('Validation Loss: {}'.format(losses.avg))
            
    def convert_mat_to_txt(self, anno_path, gt_path):
        annot = io.loadmat(anno_path)
        f = open(gt_path, 'w');
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            wr = "";
            for i in range(pts.shape[0]):
                wr += str(pts[i][0]) + " " + str(pts[i][1]) + " ";

            wr += text + " " + ori + "\n";
            f.write(wr);
        f.close()
            
    def Convert_Mat_To_Txt(self, mat_anno_folder=None, output_anno_folder=None):
        if(not os.path.isdir(output_anno_folder)):
            os.mkdir(output_anno_folder);
        else:
            os.system("rm -r " + output_anno_folder);
            os.mkdir(output_anno_folder);
            
        gt_list = os.listdir(mat_anno_folder)
        for i in tqdm(range(len(gt_list))):
            input_name = mat_anno_folder + "/" + gt_list[i];
            output_name = output_anno_folder + "/" + gt_list[i].split(".")[0] + ".txt";

            self.convert_mat_to_txt(input_name, output_name)
            
            
    def Convert_Json_To_Txt(self, img_folder=None, json_anno_file=None, output_anno_folder=None, output_img_folder=None):
        if(not os.path.isdir(output_anno_folder)):
            os.mkdir(output_anno_folder);
        else:
            os.system("rm -r " + output_anno_folder);
            os.mkdir(output_anno_folder);
            
        if(not os.path.isdir(output_img_folder)):
            os.mkdir(output_img_folder);
        else:
            os.system("rm -r " + output_img_folder);
            os.mkdir(output_img_folder);
            
        with open(json_anno_file) as json_file:
            data = json.load(json_file)
            
        anno_key_list = list(data["anns"].keys())
        img_key_list = list(data["imgs"].keys())
        complete_img_id_list = [];
        complete_img_name_list = [];
        complete_anno_list = [];
        for i in tqdm(range(len(img_key_list))):
            imgs = data["imgs"][img_key_list[i]];
            complete_img_id_list.append(imgs['id']);
            complete_img_name_list.append(imgs['file_name'])
            complete_anno_list.append([])
        
        for i in tqdm(range(len(anno_key_list))):
            anno = data["anns"][anno_key_list[i]];
            image_id = anno["image_id"];
            mask = anno["mask"];
            utf8_string = anno["utf8_string"];
            class_name = anno["class"];
            language = anno["language"];

            index = complete_img_id_list.index(image_id)
            complete_anno_list[index].append(mask)
            
        for i in tqdm(range(len(complete_anno_list))):
            anno = complete_anno_list[i];
            img_name = complete_img_name_list[i];

            if(len(anno) > 0):
                os.system("cp " + img_folder + "/" + img_name + " " + output_img_folder + "/");
                anno_file = output_anno_folder + "/" + img_name.split(".")[0] + ".txt";
                f = open(anno_file, 'w');
                #print(anno, os.path.isfile("train2014/" + img_name))
                for j in range(len(anno)):
                    tmp = anno[j];
                    wr = "";
                    for k in range(len(tmp)//2):
                        wr += str(int(tmp[k*2])) + " " + str(int(tmp[k*2+1])) + " ";

                    wr += "# # \n";
                    f.write(wr);
                f.close();
        
    
            
    # monk-text
    # mpii-mat
    # coco-json
    def Train_Dataset(self, img_folder, anno_folder, annotation_type="text"):
        self.system_dict["params"]["train_img_folder"] = img_folder;
        self.system_dict["params"]["train_anno_folder"] = anno_folder;
        self.system_dict["params"]["annotation_type"] = annotation_type;
        #print(self.system_dict["params"]["train_img_folder"], self.system_dict["params"]["train_anno_folder"]);
    
    # monk-text
    # mpii-mat
    # coco-json
    def Val_Dataset(self, img_folder, anno_folder, annotation_type="text"):
        self.system_dict["params"]["val_img_folder"] = img_folder;
        self.system_dict["params"]["val_anno_folder"] = anno_folder;
        self.system_dict["params"]["annotation_type"] = annotation_type;
        self.system_dict["params"]["val_dataset"] = True;
        #print(self.system_dict["params"]["val_img_folder"], self.system_dict["params"]["val_anno_folder"]);
        
    def Dataset_Params(self, 
                       batch_size=2, 
                       num_workers=2, 
                       input_size=512,
                       mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       rescale=255.0,
                       input_channel=1):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
        self.system_dict["params"]["input_size"] = input_size;
        self.system_dict["params"]["means"] = mean;
        self.system_dict["params"]["stds"] = std;
        self.system_dict["params"]["rescale"] = rescale;
        self.system_dict["params"]["input_channel"] = input_channel;

    def Model_Params(self, 
                     model_type="vgg", 
                     use_pretrained=True,
                     use_gpu=True,
                     use_distributed=False):
        self.system_dict["params"]["net"] = model_type;
        
        if(use_pretrained):
            self.system_dict["params"]["resume"] = "textsnake_vgg_180.pth";
            if(not os.path.isfile("textsnake_vgg_180.pth")):
                print("Downloading Model ...");
                os.system("wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YvsuxKH9M-Gseur9gc-SZJb3pCpTUddi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YvsuxKH9M-Gseur9gc-SZJb3pCpTUddi\" -O textsnake_vgg_180.pth && rm -rf /tmp/cookies.txt");
                print("Done");
        else:
            self.system_dict["params"]["resume"] = None;
        self.system_dict["params"]["cuda"] = use_gpu;
        self.system_dict["params"]["mgpu"] = use_distributed;
        self.system_dict["params"]["checkepoch"] = -1;
    
    #sgd
    #adam
    def Hyper_Params(self, 
                     optimizer="sgd",
                     lr=0.0001,
                     weight_decay=0,
                     gamma=0.1,
                     momentum=0.9):
        
        self.system_dict["params"]["optim"] = optimizer;
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        self.system_dict["params"]["gamma"] = gamma;
        self.system_dict["params"]["momentum"] = momentum;
        
        
    def Training_Params(self,
                        epochs=200,
                        output_dir="trained_weights",
                        experiment_name="exp",
                        save_freq=20,
                        display_freq=50,
                        vis_dir="vis",
                        viz_freq=50,
                        log_dir="logs",
                        log_freq=100):
        
        self.system_dict["params"]["max_epoch"] = epochs;
        self.system_dict["params"]["log_dir"] = log_dir;
        self.system_dict["params"]["exp_name"] = experiment_name;
        self.system_dict["params"]["save_dir"] = output_dir;
        self.system_dict["params"]["vis_dir"] = vis_dir;
        self.system_dict["params"]["display_freq"] = display_freq;
        self.system_dict["params"]["viz_freq"] = viz_freq;
        self.system_dict["params"]["save_freq"] = save_freq;
        self.system_dict["params"]["log_freq"] = log_freq;
        
        
    
    def Train(self):
        cfg.max_epoch = self.system_dict["params"]["max_epoch"]
        cfg.means = self.system_dict["params"]["means"]
        cfg.stds = self.system_dict["params"]["stds"]
        cfg.log_dir = self.system_dict["params"]["log_dir"]
        cfg.exp_name = self.system_dict["params"]["exp_name"]
        cfg.net = self.system_dict["params"]["net"]
        cfg.resume = self.system_dict["params"]["resume"]
        cfg.cuda = self.system_dict["params"]["cuda"]
        cfg.mgpu = self.system_dict["params"]["mgpu"]
        cfg.save_dir = self.system_dict["params"]["save_dir"]
        cfg.vis_dir = self.system_dict["params"]["vis_dir"]
        cfg.input_channel = self.system_dict["params"]["input_channel"];

       
        cfg.lr = self.system_dict["params"]["lr"];
        cfg.weight_decay = self.system_dict["params"]["weight_decay"];
        cfg.gamma = self.system_dict["params"]["gamma"];
        cfg.momentum = self.system_dict["params"]["momentum"];
        cfg.optim = self.system_dict["params"]["optim"];
        cfg.display_freq = self.system_dict["params"]["display_freq"];
        cfg.viz_freq = self.system_dict["params"]["viz_freq"];
        cfg.save_freq = self.system_dict["params"]["save_freq"];
        cfg.log_freq = self.system_dict["params"]["log_freq"];

        cfg.batch_size = self.system_dict["params"]["batch_size"];
        cfg.rescale = self.system_dict["params"]["rescale"];
        cfg.checkepoch = self.system_dict["params"]["checkepoch"];
        
        
        cfg.val_freq = 1000;
        cfg.start_iter = 0;
        cfg.loss = "CrossEntropyLoss";
        cfg.pretrain = False;
        cfg.verbose = True;
        cfg.viz = True;
        cfg.lr_adjust = "step" #fix, step
        cfg.stepvalues = [];
        cfg.step_size = cfg.max_epoch//2;
        
        
        if cfg.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
            cfg.device = torch.device("cuda")
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            cfg.device = torch.device("cpu")

        # Create weights saving directory
        if not os.path.exists(cfg.save_dir):
            os.mkdir(cfg.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(cfg.save_dir, cfg.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        if not os.path.exists(cfg.vis_dir):
            os.mkdir(cfg.vis_dir)
            
            
        
            
        if(self.system_dict["params"]["annotation_type"] == "text"):
            trainset = TotalText_txt(
                                self.system_dict["params"]["train_img_folder"], 
                                self.system_dict["params"]["train_anno_folder"],
                                ignore_list=None,
                                is_training=True,
                                transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
                            )
            train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
            
            if(self.system_dict["params"]["val_dataset"]):
                valset = TotalText_txt(
                                self.system_dict["params"]["val_img_folder"], 
                                self.system_dict["params"]["val_anno_folder"],
                                ignore_list=None,
                                is_training=False,
                                transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
                            )
                val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
            else:
                valset=None;
                
        elif(self.system_dict["params"]["annotation_type"] == "mat"):
            trainset = TotalText_mat(
                                self.system_dict["params"]["train_img_folder"], 
                                self.system_dict["params"]["train_anno_folder"],
                                ignore_list=None,
                                is_training=True,
                                transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
                            )
            train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
            
            if(self.system_dict["params"]["val_dataset"]):
                valset = TotalText_mat(
                                self.system_dict["params"]["val_img_folder"], 
                                self.system_dict["params"]["val_anno_folder"],
                                ignore_list=None,
                                is_training=False,
                                transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
                            )
                val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
            else:
                valset=None;
                
        log_dir = os.path.join(cfg.log_dir, datetime.now().strftime('%b%d_%H-%M-%S_') + cfg.exp_name)
        logger = LogSummary(log_dir)

        model = TextNet(is_training=True, backbone=cfg.net)
        if cfg.mgpu:
            model = nn.DataParallel(model)

        model = model.to(cfg.device)
        
        
        if cfg.cuda:
            cudnn.benchmark = True

        if cfg.resume:
            self.load_model(model, cfg.resume)
            
        criterion = TextLoss()
        lr = cfg.lr
        if(cfg.optim == "adam"):
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        
        train_step=0;
        print('Start training TextSnake.')
        for epoch in range(cfg.start_epoch, cfg.max_epoch):
            train_step = self.train(model, train_loader, criterion, scheduler, optimizer, epoch, logger, train_step)
            if valset:
                self.validation(model, val_loader, criterion, epoch, logger)
            self.save_model(model, "final", scheduler.get_lr(), optimizer)
                
        
    
        
                     

        
        
        
        
        
        
