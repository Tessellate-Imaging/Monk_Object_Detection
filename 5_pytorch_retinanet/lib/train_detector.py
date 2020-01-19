import collections
import os

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'


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
        self.system_dict["params"]["batch_size"] = 8;
        self.system_dict["params"]["num_workers"] = 3;
        self.system_dict["params"]["use_gpu"] = True;
        self.system_dict["params"]["lr"] = 0.0001;
        self.system_dict["params"]["gpu_devices"] = [0];
        self.system_dict["params"]["num_epochs"] = 10;
        self.system_dict["params"]["val_interval"] = 1;
        self.system_dict["params"]["print_interval"] = 20;


        self.system_dict["output"] = {};
        self.system_dict["output"]["saved_model"] = "final_model.pt";


    def Train_Dataset(self, root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=512, use_gpu=True, num_workers=3):
        self.system_dict["dataset"]["train"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["train"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["set_dir"] = set_dir;


        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["image_size"] = image_size;
        self.system_dict["params"]["use_gpu"] = use_gpu;
        self.system_dict["params"]["num_workers"] = num_workers;


        self.system_dict["local"]["dataset_train"] = CocoDataset(self.system_dict["dataset"]["train"]["root_dir"] + "/" + self.system_dict["dataset"]["train"]["coco_dir"], 
                                                            img_dir=self.system_dict["dataset"]["train"]["img_dir"], 
                                                            set_dir=self.system_dict["dataset"]["train"]["set_dir"],
                                                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        self.system_dict["local"]["sampler"] = AspectRatioBasedSampler(self.system_dict["local"]["dataset_train"], 
                                                                    batch_size=self.system_dict["params"]["batch_size"], drop_last=False)
        
        self.system_dict["local"]["dataloader_train"] = DataLoader(self.system_dict["local"]["dataset_train"], 
                                                                num_workers=self.system_dict["params"]["num_workers"], 
                                                                collate_fn=collater, 
                                                                batch_sampler=self.system_dict["local"]["sampler"])

        print('Num training images: {}'.format(len(self.system_dict["local"]["dataset_train"])))


    def Val_Dataset(self, root_dir, coco_dir, img_dir, set_dir):
        self.system_dict["dataset"]["val"]["status"] = True;
        self.system_dict["dataset"]["val"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["val"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["set_dir"] = set_dir;  


        self.system_dict["local"]["dataset_val"] = CocoDataset(self.system_dict["dataset"]["val"]["root_dir"] + "/" + self.system_dict["dataset"]["val"]["coco_dir"], 
                                                            img_dir=self.system_dict["dataset"]["val"]["img_dir"], 
                                                            set_dir=self.system_dict["dataset"]["val"]["set_dir"],
                                                            transform=transforms.Compose([Normalizer(), Resizer()]))

        self.system_dict["local"]["sampler_val"] = AspectRatioBasedSampler(self.system_dict["local"]["dataset_val"], 
                                                                    batch_size=self.system_dict["params"]["batch_size"], drop_last=False)


        self.system_dict["local"]["dataloader_val"] = DataLoader(self.system_dict["local"]["dataset_val"], 
                                                                num_workers=self.system_dict["params"]["num_workers"], 
                                                                collate_fn=collater, 
                                                                batch_sampler=self.system_dict["local"]["sampler_val"])

        print('Num validation images: {}'.format(len(self.system_dict["local"]["dataset_val"])))


    def Model(self, model_name="resnet18",gpu_devices=[0]):

        num_classes = self.system_dict["local"]["dataset_train"].num_classes();
        if model_name == "resnet18":
            retinanet = model.resnet18(num_classes=num_classes, pretrained=True)
        elif model_name == "resnet34":
            retinanet = model.resnet34(num_classes=num_classes, pretrained=True)
        elif model_name == "resnet50":
            retinanet = model.resnet50(num_classes=num_classes, pretrained=True)
        elif model_name == "resnet101":
            retinanet = model.resnet101(num_classes=num_classes, pretrained=True)
        elif model_name == "resnet152":
            retinanet = model.resnet152(num_classes=num_classes, pretrained=True)

        if self.system_dict["params"]["use_gpu"]:
            self.system_dict["params"]["gpu_devices"] = gpu_devices
            if len(self.system_dict["params"]["gpu_devices"])==1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.system_dict["params"]["gpu_devices"][0])
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in self.system_dict["params"]["gpu_devices"]])
            self.system_dict["local"]["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
            retinanet = retinanet.to(self.system_dict["local"]["device"])
            retinanet = torch.nn.DataParallel(retinanet).to(self.system_dict["local"]["device"])

        retinanet.training = True
        retinanet.train()
        retinanet.module.freeze_bn()

        self.system_dict["local"]["model"] = retinanet;



    def Set_Hyperparams(self, lr=0.0001, val_interval=1, print_interval=20):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["val_interval"] = val_interval;
        self.system_dict["params"]["print_interval"] = print_interval;


        self.system_dict["local"]["optimizer"] = torch.optim.Adam(self.system_dict["local"]["model"].parameters(), 
                                                                    self.system_dict["params"]["lr"]);

        self.system_dict["local"]["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.system_dict["local"]["optimizer"], 
                                                                    patience=3, verbose=True)

        self.system_dict["local"]["loss_hist"] = collections.deque(maxlen=500)


    def Train(self, num_epochs=2, output_model_name="final_model.pt"):
        self.system_dict["output"]["saved_model"] = output_model_name;
        self.system_dict["params"]["num_epochs"] = num_epochs;

        for epoch_num in range(num_epochs):
            self.system_dict["local"]["model"].train()
            self.system_dict["local"]["model"].module.freeze_bn()

            epoch_loss = []

            for iter_num, data in enumerate(self.system_dict["local"]["dataloader_train"]):
                try:
                    self.system_dict["local"]["optimizer"].zero_grad()

                    classification_loss, regression_loss = self.system_dict["local"]["model"]([data['img'].to(self.system_dict["local"]["device"]).float(),  data['annot'].to(self.system_dict["local"]["device"])])

                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                    if bool(loss == 0):
                        continue

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.system_dict["local"]["model"].parameters(), 0.1)

                    self.system_dict["local"]["optimizer"].step()

                    self.system_dict["local"]["loss_hist"].append(float(loss))

                    epoch_loss.append(float(loss))
                    
                    if(iter_num % self.system_dict["params"]["print_interval"] == 0):
                        print(
                            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(self.system_dict["local"]["loss_hist"])))

                    del classification_loss
                    del regression_loss

                except Exception as e:
                    print(e)
                    continue
            
            if(self.system_dict["dataset"]["val"]["status"]):        
                print('Evaluating dataset')
                coco_eval.evaluate_coco(self.system_dict["local"]["dataset_val"], self.system_dict["local"]["model"])
            
            self.system_dict["local"]["scheduler"].step(np.mean(epoch_loss))

            torch.save(self.system_dict["local"]["model"], 'resume.pt')

        self.system_dict["local"]["model"].eval()

        torch.save(self.system_dict["local"]["model"], output_model_name)
