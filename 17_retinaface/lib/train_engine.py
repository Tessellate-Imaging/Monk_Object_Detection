from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data.wider_face import WiderFaceDetection, DataLoaderWithoutLandmarks, detection_collate
from data.config import cfg_mnet, cfg_re50
from data.data_augment import preproc
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace


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
        
    def Train_Dataset(self, img_dir, anno_file):
        self.system_dict["params"]["img_dir"] = img_dir;
        self.system_dict["params"]["anno_file"] = anno_file;
        
    def Dataset_Params(self, batch_size=32, num_workers=4):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;
        
    def List_Models(self):
        self.system_dict["params"]["model_list"] = ["mobilenet", "resnet"]
        
        for i in range(len(self.system_dict["params"]["model_list"])):
            print("{}. Model - {}".format(i+1, self.system_dict["params"]["model_list"][i]));
        
    def Model_Params(self, model_type="mobilenet", use_gpu=True, devices=[0], resume_from=None):
        if(model_type == "mobilenet"):
            self.system_dict["params"]["network"] = "mobile0.25";
        elif(model_type == "resnet"):
            self.system_dict["params"]["network"] = "resnet50";
        else:
            print("Model type not found");
            print("Available models - mobilenet, resnet");
        self.system_dict["params"]["use_gpu"] = use_gpu;
        self.system_dict["params"]["resume_net"] = resume_from;
        self.system_dict["params"]["resume_epoch"] = 0;
        self.system_dict["params"]["num_gpu"] = len(devices);
        
    def Hyper_Parameters(self, lr=0.0001, momentum=0.9, weight_decay=0.0005, gamma=0.1):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;
        self.system_dict["params"]["gamma"] = gamma;
        
    def Training_Params(self, num_epochs=2, output_dir="weights_trained"):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["save_folder"] = output_dir;
    
    def Train(self):
        self.setup();
        cfg = self.system_dict["local"]["cfg"];
        print(cfg);
        
        rgb_mean = (104, 117, 123) # bgr order
        num_classes = 2
        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        max_epoch = cfg['epoch']
        gpu_train = cfg['gpu_train']
        
        num_workers = self.system_dict["params"]["num_workers"]
        momentum = self.system_dict["params"]["momentum"]
        weight_decay = self.system_dict["params"]["weight_decay"]
        initial_lr = self.system_dict["params"]["lr"]
        gamma = self.system_dict["params"]["gamma"]
        save_folder = self.system_dict["params"]["save_folder"]
        
        print("Loading Network...");
        net = RetinaFace(cfg=cfg)

        if self.system_dict["params"]["resume_net"] is not None:
            print('Loading resume network...')
            state_dict = torch.load(self.system_dict["params"]["resume_net"])
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

        if num_gpu > 1 and gpu_train:
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = net.cuda()
        cudnn.benchmark = True
        print("Done...");


        optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

        priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
        with torch.no_grad():
            priors = priorbox.forward()
            priors = priors.cuda()
            
        net.train()
        epoch = 0 + self.system_dict["params"]["resume_epoch"]
        dataset = self.system_dict["local"]["dataset"];
        
        epoch_size = math.ceil(len(dataset) / batch_size)
        max_iter = max_epoch * epoch_size

        stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
        step_index = 0

        if self.system_dict["params"]["resume_epoch"] > 0:
            start_iter = self.system_dict["params"]["resume_epoch"] * epoch_size
        else:
            start_iter = 0

        for iteration in range(start_iter, max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
                torch.save(net.state_dict(), save_folder + "/" + cfg['name']+ '_intermediate.pth')
                epoch += 1

            load_t0 = time.time()
            if iteration in stepvalues:
                step_index += 1
            lr = self.adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, initial_lr)

            # load train data
            images, targets = next(batch_iterator)
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if(iteration % 50 == 0):
                print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                      .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

        torch.save(net.state_dict(), save_folder + "/" + cfg['name'] + '_Final.pth')
        

        
        
        
    def setup(self):
        if not os.path.exists(self.system_dict["params"]["save_folder"]):
            os.mkdir(self.system_dict["params"]["save_folder"])
        self.system_dict["local"]["cfg"] = None
        if self.system_dict["params"]["network"] == "mobile0.25":
            self.system_dict["local"]["cfg"] = cfg_mnet
        elif self.system_dict["params"]["network"] == "resnet50":
            self.system_dict["local"]["cfg"] = cfg_re50
        
        self.system_dict["local"]["cfg"]["gpu_train"] = self.system_dict["params"]["use_gpu"];
        self.system_dict["local"]["cfg"]["batch_size"] = self.system_dict["params"]["batch_size"];
        self.system_dict["local"]["cfg"]["ngpu"] = self.system_dict["params"]["num_gpu"];
        self.system_dict["local"]["cfg"]["epoch"] = self.system_dict["params"]["num_epochs"];
        
        rgb_mean = (104, 117, 123) # bgr order
        img_dim = self.system_dict["local"]["cfg"]['image_size'];
        print('Loading Dataset...')
        self.system_dict["local"]["dataset"] = DataLoaderWithoutLandmarks(self.system_dict["params"]["img_dir"], 
                                                                          self.system_dict["params"]["anno_file"], 
                                                                          preproc(img_dim, rgb_mean))
        print("Done...");
        
        
        
    
    def adjust_learning_rate(self, optimizer, gamma, epoch, step_index, iteration, epoch_size, initial_lr):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warmup_epoch = -1
        if epoch <= warmup_epoch:
            lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
        else:
            lr = initial_lr * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
                                                   
                                                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    