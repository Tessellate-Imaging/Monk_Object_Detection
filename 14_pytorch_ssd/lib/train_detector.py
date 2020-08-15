import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform


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
        
        self.set_base_params();
        
    def set_base_params(self):
        self.system_dict["params"]["dataset_type"] = "voc";
        self.system_dict["params"]["balance_data"] = False;
        self.system_dict["params"]["label_file"] = None;
        self.system_dict["params"]["batch_size"] = 32;
        self.system_dict["params"]["num_workers"] = 4;
        
        self.system_dict["params"]["net"] = "mb1-ssd"; #mb1-ssd, mb2-ssd-lite, vgg16-ssd
        self.system_dict["params"]["freeze_base_net"] = False;
        self.system_dict["params"]["freeze_net"] = False;
        self.system_dict["params"]["mb2_width_mult"] = 1.0;
        
        self.system_dict["params"]["base_net"] = None;
        self.system_dict["params"]["resume"] = None;
        self.system_dict["params"]["pretrained_ssd"] = None;
        self.system_dict["params"]["use_cuda"] = True;
        
        self.system_dict["params"]["lr"] = 0.001;
        self.system_dict["params"]["momentum"] = 0.09
        self.system_dict["params"]["weight_decay"] = 0.0005;
        self.system_dict["params"]["gamma"] = 0.1;
        self.system_dict["params"]["base_net_lr"] = None;
        self.system_dict["params"]["extra_layers_lr"] = None;
        
        self.system_dict["params"]["scheduler"] = "multi-step"; #cosine
        self.system_dict["params"]["milestones"] = "80,100";
        self.system_dict["params"]["t_max"] = 120;
        
        self.system_dict["params"]["checkpoint_folder"] = "models/"
        self.system_dict["params"]["num_epochs"] = 120;
        self.system_dict["params"]["validation_epochs"] = 5;
        self.system_dict["params"]["debug_steps"] = 100;
        
    def set_train_data_params(self, img_dir, label_dir, label_file, batch_size=2, balance_data=False, num_workers=4):
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["label_dir"] = label_dir;
        self.system_dict["params"]["label_file"] = label_file;
        
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["balance_data"] = balance_data;
        self.system_dict["params"]["num_workers"] = num_workers;
        
    def set_val_data_params(self, img_dir, label_dir):
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["label_dir"] = label_dir;
        self.system_dict["dataset"]["val"]["status"] = True;
        
    def set_model_params(self, net="mb1-ssd", freeze_base_net=False, 
                         freeze_net=False, use_gpu=True, resume=False, mb2_width_mult=1.0):
        self.system_dict["params"]["net"] = net;
        self.system_dict["params"]["freeze_net"] = freeze_net;
        self.system_dict["params"]["freeze_base_net"] = freeze_base_net;
        self.system_dict["params"]["mb2_width_mult"] = mb2_width_mult;
        
        self.system_dict["params"]["resume"] = resume;
        self.system_dict["params"]["use_cuda"] = use_gpu;
        
        print("Downloading model");
        if(net == "mb1-ssd"):
            if(not os.path.isfile("mobilenet-v1-ssd-mp-0_675.pth")):
                os.system("wget https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth");
            self.system_dict["params"]["pretrained_ssd"] = "mobilenet-v1-ssd-mp-0_675.pth";
        elif(net == "mb2-ssd-lite"):
            if(not os.path.isfile("mb2-ssd-lite-mp-0_686.pth")):
                os.system("wget https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth");
            self.system_dict["params"]["pretrained_ssd"] = "mb2-ssd-lite-mp-0_686.pth";
        elif(net == "vgg16-ssd"):
            if(not os.path.isfile("vgg16-ssd-mp-0_7726")):
                os.system("https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth");
            self.system_dict["params"]["pretrained_ssd"] = "vgg16-ssd-mp-0_7726.pth"; 
        print("Model downloaded");
        
    def set_lr_params(self, lr=0.001, base_net_lr=None, extra_layers_lr=None,
                      scheduler="multi-step", milestones=None, t_max=120, gamma=0.1):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["base_net_lr"] = base_net_lr;
        self.system_dict["params"]["extra_layers_lr"] = extra_layers_lr;
        self.system_dict["params"]["scheduler"] = scheduler
        self.system_dict["params"]["milestones"] = milestones;
        self.system_dict["params"]["t_max"] = t_max;
        self.system_dict["params"]["gamma"] = gamma;
        
    def set_optimizer_params(self, momentum=0.09, weight_decay=0.0005):
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        
    def train(self, num_epochs=5, val_epoch_interval=2, output_folder="models_dir/", debug_steps=100):
        self.system_dict["params"]["checkpoint_folder"] = output_folder
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["validation_epochs"] = val_epoch_interval;
        self.system_dict["params"]["debug_steps"] = debug_steps;
        
        if(not os.path.isdir(self.system_dict["params"]["checkpoint_folder"])):
            os.mkdir(self.system_dict["params"]["checkpoint_folder"]);
        
        self.setup_and_start_training();
        
    def setup_and_start_training(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and self.system_dict["params"]["use_cuda"] else "cpu")
        
        if self.system_dict["params"]["use_cuda"] and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logging.info("Using gpu.");
        else:
            logging.info("Using cpu.");
        
        timer = Timer()
        logging.info(self.system_dict);
        
        if self.system_dict["params"]["net"] == 'vgg16-ssd':
            create_net = create_vgg_ssd
            config = vgg_ssd_config
        elif self.system_dict["params"]["net"] == 'mb1-ssd':
            create_net = create_mobilenetv1_ssd
            config = mobilenetv1_ssd_config
        elif self.system_dict["params"]["net"] == 'mb1-ssd-lite':
            create_net = create_mobilenetv1_ssd_lite
            config = mobilenetv1_ssd_config
        elif self.system_dict["params"]["net"] == 'sq-ssd-lite':
            create_net = create_squeezenet_ssd_lite
            config = squeezenet_ssd_config
        elif self.system_dict["params"]["net"] == 'mb2-ssd-lite':
            create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=self.system_dict["params"]["mb2_width_mult"])
            config = mobilenetv1_ssd_config
        else:
            logging.fatal("The net type is wrong.")
            sys.exit(1)
            
        train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
        target_transform = MatchPrior(config.priors, config.center_variance,
                                      config.size_variance, 0.5)

        test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

        logging.info("Prepare training datasets.")
        datasets = [];
        dataset = VOCDataset(self.system_dict["dataset"]["val"]["img_dir"], 
                             self.system_dict["dataset"]["val"]["label_dir"],
                             transform=train_transform,
                             target_transform=target_transform,
                             label_file=self.system_dict["params"]["label_file"])
        label_file = self.system_dict["params"]["label_file"]
        #store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)
        logging.info(f"Stored labels into file {label_file}.")
        train_dataset = ConcatDataset(datasets)
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, self.system_dict["params"]["batch_size"],
                                  num_workers=self.system_dict["params"]["num_workers"],
                                  shuffle=True)
        
        if(self.system_dict["dataset"]["val"]["status"]):
            val_dataset = VOCDataset(self.system_dict["dataset"]["val"]["img_dir"], 
                                     self.system_dict["dataset"]["val"]["label_dir"], 
                                     transform=test_transform,
                                     target_transform=target_transform, 
                                     is_test=True,
                                     label_file=self.system_dict["params"]["label_file"])
            logging.info("validation dataset size: {}".format(len(val_dataset)))
            val_loader = DataLoader(val_dataset, self.system_dict["params"]["batch_size"],
                            num_workers=self.system_dict["params"]["num_workers"],
                            shuffle=False)
        
        
        logging.info("Build network.")
        net = create_net(num_classes)
        min_loss = -10000.0
        last_epoch = -1

        base_net_lr = self.system_dict["params"]["base_net_lr"] if self.system_dict["params"]["base_net_lr"] is not None else self.system_dict["params"]["lr"]
        extra_layers_lr = self.system_dict["params"]["extra_layers_lr"] if self.system_dict["params"]["extra_layers_lr"] is not None else self.system_dict["params"]["lr"]
        
        if self.system_dict["params"]["freeze_base_net"]:
            logging.info("Freeze base net.")
            freeze_net_layers(net.base_net)
            params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                     net.regression_headers.parameters(), net.classification_headers.parameters())
            params = [
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]
        elif self.system_dict["params"]["freeze_net"]:
            freeze_net_layers(net.base_net)
            freeze_net_layers(net.source_layer_add_ons)
            freeze_net_layers(net.extras)
            params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
            logging.info("Freeze all the layers except prediction heads.")
        else:
            params = [
                {'params': net.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]
        
        timer.start("Load Model")
        resume = self.system_dict["params"]["resume"];
        base_net = self.system_dict["params"]["base_net"]
        pretrained_ssd = self.system_dict["params"]["pretrained_ssd"];
        if self.system_dict["params"]["resume"]:
            logging.info(f"Resume from the model {resume}")
            net.load(self.system_dict["params"]["resume"])
        elif self.system_dict["params"]["base_net"]:
            logging.info(f"Init from base net {base_net}")
            net.init_from_base_net(self.system_dict["params"]["base_net"])
        elif self.system_dict["params"]["pretrained_ssd"]:
            logging.info(f"Init from pretrained ssd {pretrained_ssd}")
            net.init_from_pretrained_ssd(self.system_dict["params"]["pretrained_ssd"])
        logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

        net.to(DEVICE)   
        
        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
        optimizer = torch.optim.SGD(params, 
                                    lr=self.system_dict["params"]["lr"], 
                                    momentum=self.system_dict["params"]["momentum"],
                                    weight_decay=self.system_dict["params"]["weight_decay"])
        lr = self.system_dict["params"]["lr"];
        logging.info(f"Learning rate: {lr}, Base net learning rate: {base_net_lr}, "
                     + f"Extra Layers learning rate: {extra_layers_lr}.")
        
        if(not self.system_dict["params"]["milestones"]):
            self.system_dict["params"]["milestones"] = "";
            self.system_dict["params"]["milestones"] += str(int(self.system_dict["params"]["num_epochs"]/3)) + ",";
            self.system_dict["params"]["milestones"] += str(int(2*self.system_dict["params"]["num_epochs"]/3));
            
        if self.system_dict["params"]["scheduler"] == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in self.system_dict["params"]["milestones"].split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                         gamma=0.1, last_epoch=last_epoch)
        elif self.system_dict["params"]["scheduler"] == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, self.system_dict["params"]["t_max"], last_epoch=last_epoch)
        
        
        logging.info(f"Start training from epoch {last_epoch + 1}.")
        for epoch in range(last_epoch + 1, self.system_dict["params"]["num_epochs"]):
            scheduler.step()
            self.base_train(train_loader, net, criterion, optimizer,
                  device=DEVICE, debug_steps=self.system_dict["params"]["debug_steps"], epoch=epoch)

            if((self.system_dict["dataset"]["val"]["status"]) and (epoch % self.system_dict["params"]["validation_epochs"] == 0 or epoch == self.system_dict["params"]["num_epochs"] - 1)):
                val_loss, val_regression_loss, val_classification_loss = self.base_test(val_loader, net, criterion, DEVICE)
                logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
                net_name = self.system_dict["params"]["net"];
                model_path = os.path.join(self.system_dict["params"]["checkpoint_folder"], f"{net_name}-Epoch-{epoch}-Loss-{val_loss}.pth")
                net.save(model_path)
                logging.info(f"Saved model {model_path}")
            if(not self.system_dict["dataset"]["val"]["status"]):
                model_path = os.path.join(self.system_dict["params"]["checkpoint_folder"], f"{net_name}-Epoch-{epoch}.pth")
                net.save(model_path)
                logging.info(f"Saved model {model_path}")
            
            
            
        
    def base_train(self, loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
        net.train(True)
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        for i, data in enumerate(loader):
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            if i and i % debug_steps == 0:
                avg_loss = running_loss / debug_steps
                avg_reg_loss = running_regression_loss / debug_steps
                avg_clf_loss = running_classification_loss / debug_steps
                logging.info(
                    f"Epoch: {epoch}, Step: {i}, " +
                    f"Average Loss: {avg_loss:.4f}, " +
                    f"Average Regression Loss {avg_reg_loss:.4f}, " +
                    f"Average Classification Loss: {avg_clf_loss:.4f}"
                )
                running_loss = 0.0
                running_regression_loss = 0.0
                running_classification_loss = 0.0


    def base_test(self, loader, net, criterion, device):
        net.eval()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        num = 0
        for _, data in enumerate(loader):
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            num += 1

            with torch.no_grad():
                confidence, locations = net(images)
                regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                loss = regression_loss + classification_loss

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
        return running_loss / num, running_regression_loss / num, running_classification_loss / num
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        