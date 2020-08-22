import os
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import shutil
import argparse
from peleenet import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *
from tqdm import tqdm

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

    def Train_Dataset(self, img_dir, anno_dir, class_file):
        self.system_dict["params"]["dataset"] = "VOC";
        self.system_dict["params"]["train_img_dir"] = img_dir;
        self.system_dict["params"]["train_anno_dir"] = anno_dir;
        self.system_dict["params"]["class_file"] = class_file;

    def Dataset_Params(self, batch_size=2, num_workers=2):
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["num_workers"] = num_workers;

    def Model_Params(self, gpu_devices=[0], resume_from=None, resume_epoch=0):
        if(not resume_from):
            if(not os.path.isdir("weights")):
                os.mkdir("weights");
            if(not os.path.isfile("weights/Pelee_VOC.pth")):
                os.system("wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16HparGAVhxTDByi5RylYCkxLZYducK9j' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16HparGAVhxTDByi5RylYCkxLZYducK9j\" -O Pelee_VOC.pth && rm -rf /tmp/cookies.txt");
                os.system("mv Pelee_VOC.pth weights/");
            self.system_dict["params"]["resume_net"] = None;
            self.system_dict["params"]["resume_epoch"] = 0;
        else:
            self.system_dict["params"]["resume_net"] = resume_from;
        self.system_dict["params"]["resume_epoch"] = resume_epoch;
        if(len(gpu_devices) > 0):
            self.system_dict["params"]["use_gpu"] = True;
        else:
            self.system_dict["params"]["use_gpu"] = False;
        self.system_dict["params"]["ngpu"] = len(gpu_devices);

    def Hyper_Params(self, lr=0.01, gamma=0.1, momentum=0.9, weight_decay=0.0005):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["gamma"] = gamma;
        self.system_dict["params"]["momentum"] = momentum;
        self.system_dict["params"]["weight_decay"] = weight_decay;

    def Training_Params(self, num_epochs=2, model_output_dir="output"):
        self.system_dict["params"]["num_epochs"] = num_epochs;
        self.system_dict["params"]["model_output_dir"] = model_output_dir;

    def Train(self):
        self.setup();

        
        print_info('===> Training STDN', ['yellow', 'bold'])
        epoch = self.system_dict["params"]["resume_epoch"]
        self.system_dict["local"]["start_iter"] = self.system_dict["[arams]"]["resume_epoch"] * self.system_dict["local"]["epoch_size"] if self.system_dict["params"]["resume_epoch"]  > 0 else 0
        step_index = 0
        for step in self.system_dict["local"]["stepvalues"]:
            if self.system_dict["local"]["start_iter"] > step:
                step_index += 1

        for iteration in range(self.system_dict["local"]["start_iter"], self.system_dict["local"]["max_iter"]):
            if iteration % self.system_dict["local"]["epoch_size"] == 0:
                batch_iterator = iter(data.DataLoader(self.system_dict["local"]["dataset"],
                                                      self.system_dict["local"]["cfg"].train_cfg.per_batch_size * self.system_dict["params"]["ngpu"],
                                                      shuffle=True,
                                                      num_workers=self.system_dict["local"]["cfg"].train_cfg.num_workers,
                                                      collate_fn=detection_collate))
                if epoch % self.system_dict["local"]["cfg"].model.save_epochs == 0:
                    save_checkpoint(self.system_dict["local"]["net"], self.system_dict["local"]["cfg"], final=False,
                                    datasetname=self.system_dict["params"]["dataset"], epoch=-1)
                epoch += 1

            load_t0 = time.time()
            if iteration in self.system_dict["local"]["stepvalues"]:
                step_index += 1
            
            lr = adjust_learning_rate(self.system_dict["local"]["optimizer"], 
                                        step_index, 
                                        self.system_dict["local"]["cfg"], 
                                        self.system_dict["params"]["dataset"])
            
            images, targets = next(batch_iterator)
            if self.system_dict["local"]["cfg"].train_cfg.cuda:
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]
            out = self.system_dict["local"]["net"](images)
            self.system_dict["local"]["optimizer"].zero_grad()
            loss_l, loss_c = self.system_dict["local"]["criterion"](out, self.system_dict["local"]["priors"], targets)
            loss = loss_l + loss_c

            loss.backward()
            self.system_dict["local"]["optimizer"].step()
            load_t1 = time.time()
            if(iteration % 50 == 0 or iteration==self.system_dict["local"]["max_iter"]):
                print_train_log(iteration, 
                                self.system_dict["local"]["cfg"].train_cfg.print_epochs,
                                [time.ctime(), 
                                    epoch, 
                                    iteration % self.system_dict["local"]["epoch_size"], 
                                    self.system_dict["local"]["epoch_size"], 
                                    iteration, loss_l.item(), loss_c.item(), 
                                    load_t1 - load_t0, 
                                    self.system_dict["params"]["lr"]])

        save_checkpoint(self.system_dict["local"]["net"], 
                        self.system_dict["local"]["cfg"], 
                        final=True,
                        datasetname=self.system_dict["params"]["dataset"], epoch=-1)

        os.system("cp config_final.py " + self.system_dict["params"]["model_output_dir"] + "/")
        

    def setup(self):
        f = open("Monk_Object_Detection/15_pytorch_peleenet/lib/configs/Pelee_VOC.py");
        lines = f.read();
        f.close();

        lines = lines.replace("save_epochs=10",
                                "save_epochs=" + str(self.system_dict["params"]["num_epochs"]));
        lines = lines.replace("print_epochs=10",
                                "print_epochs=" + str(self.system_dict["params"]["num_epochs"]));
        lines = lines.replace("weights_save='weights/'",
                                "weights_save='" + self.system_dict["params"]["model_output_dir"] + "/'");
        if(self.system_dict["params"]["use_gpu"]):
            lines = lines.replace("cuda=True",
                                    "cuda=True");
        else:
            lines = lines.replace("cuda=True",
                                    "cuda=False");
        lines = lines.replace("per_batch_size=64",
                                "per_batch_size=" + str(self.system_dict["params"]["batch_size"]));
        lines = lines.replace("num_workers=8",
                                "num_workers=" + str(self.system_dict["params"]["num_workers"]));

        f = open("config.py", 'w');
        f.write(lines);
        f.close();

        self.system_dict["local"]["cfg"] = Config.fromfile("config.py");
        

        print_info('===> Loading Dataset...', ['yellow', 'bold'])
        self.system_dict["local"]["dataset"] = get_dataloader(self.system_dict["local"]["cfg"],
                                                                train_img_dir=self.system_dict["params"]["train_img_dir"], 
                                                                train_anno_dir=self.system_dict["params"]["train_anno_dir"],
                                                                class_file=self.system_dict["params"]["class_file"]);
        print_info('===> Done...', ['yellow', 'bold'])

        print_info('===> Setting up epoch details...', ['yellow', 'bold'])
        self.system_dict["local"]["epoch_size"] = len(self.system_dict["local"]["dataset"]) // (self.system_dict["local"]["cfg"].train_cfg.per_batch_size * self.system_dict["params"]["ngpu"])

        self.system_dict["local"]["max_iter"] = self.system_dict["local"]["epoch_size"]*self.system_dict["params"]["num_epochs"];

        self.system_dict["local"]["stepvalues"] = [self.system_dict["local"]["max_iter"]//3, 2*self.system_dict["local"]["max_iter"]//3];

        f = open("config.py");
        lines = f.read();
        f.close();

        lines = lines.replace("step_lr=[80000, 100000, 120000,160000]",
                                "step_lr=" + str(self.system_dict["local"]["stepvalues"]));
        lines = lines.replace("num_classes=21",
                                "num_classes=" + str(len(self.system_dict["local"]["dataset"].class_to_ind)));
        lines = lines.replace("lr=5e-3",
                                "lr=" + str(self.system_dict["params"]["lr"]));
        lines = lines.replace("gamma=0.1",
                                "gamma=" + str(self.system_dict["params"]["gamma"]));
        lines = lines.replace("momentum=0.9",
                                "momentum=" + str(self.system_dict["params"]["momentum"]));
        lines = lines.replace("weight_decay=0.0005",
                                "weight_decay=" + str(self.system_dict["params"]["weight_decay"]));


        f = open("config_final.py", 'w');
        f.write(lines);
        f.close();
        print_info('===> Done...', ['yellow', 'bold'])

        self.system_dict["local"]["cfg"] = Config.fromfile("config_final.py");
        #print(self.system_dict["local"]["cfg"])

        self.system_dict["local"]["net"] = build_net('train', 
                                                        self.system_dict["local"]["cfg"].model.input_size, 
                                                        self.system_dict["local"]["cfg"].model)
        
        if(self.system_dict["params"]["resume_net"]):
            init_net(self.system_dict["local"]["net"], 
                        self.system_dict["local"]["cfg"], 
                        self.system_dict["params"]["resume_net"])  # init the network with pretrained
        if self.system_dict["params"]["ngpu"] > 1:
            self.system_dict["local"]["net"] = torch.nn.DataParallel(self.system_dict["local"]["net"])
        if self.system_dict["local"]["cfg"].train_cfg.cuda:
            self.system_dict["local"]["net"].cuda()
            cudnn.benckmark = True

        self.system_dict["local"]["optimizer"] = set_optimizer(self.system_dict["local"]["net"], 
                                                                self.system_dict["local"]["cfg"])
        self.system_dict["local"]["criterion"] = set_criterion(self.system_dict["local"]["cfg"])
        self.system_dict["local"]["priorbox"] = PriorBox(anchors(self.system_dict["local"]["cfg"].model))

        with torch.no_grad():
            self.system_dict["local"]["priors"] = self.system_dict["local"]["priorbox"].forward()
            if self.system_dict["local"]["cfg"].train_cfg.cuda:
                self.system_dict["local"]["priors"] = self.system_dict["local"]["priors"].cuda()
        

