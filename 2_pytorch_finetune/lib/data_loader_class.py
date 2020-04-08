import os
import sys
import numpy as np
import pandas as pd
import cv2

from engine import train_one_epoch, evaluate
import utils
import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.notebook import tqdm



class CustomDatasetMultiObject(object):
    def __init__(self, root, img_dir, anno_file, transforms):
        self.root = root
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.train_list = pd.read_csv(root + "/" + anno_file);
        self.label_list = self.get_labels();
        self.num_classes = len(self.label_list) + 1;
        self.columns = self.train_list.columns
        
    def get_labels(self):
        label_list = [];
        for i in range(len(self.train_list)):
            label = self.train_list[self.columns[1]][i];
            tmp = label.split(" ");
            for j in range(len(tmp)//5):
                if(tmp[(j*5+4)] not in label_list):
                    label_list.append(tmp[(j*5+4)])
        return sorted(label_list);
        

    def __getitem__(self, idx):
        # load images ad masks

        img_name = self.train_list[self.columns[0]][idx];
        label = self.train_list[self.columns[1]][idx];
        
        
        img_path = os.path.join(self.root, self.img_dir, img_name)  
        img = Image.open(img_path).convert("RGB")
        h, w = img.size;
        tmp = label.split(" ");
        boxes = [];

        num_objs = 0;
        obj_ids = [];
        for j in range(len(tmp)//5):
            x1 = int(float(tmp[(j*5+0)]));
            y1 = int(float(tmp[(j*5+1)]));
            x2 = int(float(tmp[(j*5+2)]));
            y2 = int(float(tmp[(j*5+3)]));
            label = tmp[(j*5+4)];
            boxes.append([x1, y1, x2, y2]);
            obj_ids.append(self.label_list.index(label)+1);
            num_objs += 1;
        obj_ids = np.array(obj_ids, dtype=np.int64);
        #print(obj_ids)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        
        #print(labels)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.train_list)
