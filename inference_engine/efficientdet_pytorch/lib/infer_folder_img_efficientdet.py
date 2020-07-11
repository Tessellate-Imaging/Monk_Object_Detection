# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""

import os
import sys

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

import pandas as pd
import cv2
import json
from pascal_voc_writer import Writer



def preds_to_monk_format(img_name, preds, classes_selected, class_names=[], thresh=0.5):
    if(len(classes_selected) == 1 and classes_selected[0] == ""):
        classes_selected = class_names;
    combined = [];
    wr = "";
    i = 0;
    for j in range(len(preds[i]['rois'])):
        x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
        class_name = obj_list[preds[i]['class_ids'][j]]
        score = float(preds[i]['scores'][j])
        if(score > thresh and class_name in classes_selected):
            wr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + class_name + " ";
    
    wr = wr[:len(wr)-1];
    combined.append([img_name, wr]);
    df = pd.DataFrame(combined, columns = ['ID', 'Label'])  
    return df;


def preds_to_coco_format(img_name, preds, classes_selected, class_names=[], thresh=0.5):
    if(len(classes_selected) == 1 and classes_selected[0] == ""):
        classes_selected = class_names;
    combined = [];
    wr = "";
    i = 0;
    for j in range(len(preds[i]['rois'])):
        x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
        class_name = obj_list[preds[i]['class_ids'][j]]
        score = float(preds[i]['scores'][j])
        if(score > thresh and class_name in classes_selected):
            wr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + class_name + " ";
    
    wr = wr[:len(wr)-1];
    combined.append([img_name, wr]);
    df = pd.DataFrame(combined, columns = ['ID', 'Label'])
    
    columns = df.columns
    
    delimiter = " ";
    
    
    output_classes_file = "tmp.txt"
    list_dict = [];
    anno = [];
    for i in range(len(df)):
        img_name = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(delimiter);
        for j in range(len(tmp)//5):
            label = tmp[j*5+4];
            if(label not in anno):
                anno.append(label);
        anno = sorted(anno)

    for i in range(len(anno)):
        tmp = {};
        tmp["supercategory"] = "master";
        tmp["id"] = i;
        tmp["name"] = anno[i];
        list_dict.append(tmp);

    anno_f = open(output_classes_file, 'w');
    for i in range(len(anno)):
        anno_f.write(anno[i] + "\n");
    anno_f.close();

    
    coco_data = {};
    coco_data["type"] = "instances";
    coco_data["images"] = [];
    coco_data["annotations"] = [];
    coco_data["categories"] = list_dict;
    image_id = 0;
    annotation_id = 0;


    for i in range(len(df)):
        img_name = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(delimiter);
        image_in_path = img_name;
        img = cv2.imread(image_in_path, 1);
        h, w, c = img.shape;

        images_tmp = {};
        images_tmp["file_name"] = img_name;
        images_tmp["height"] = h;
        images_tmp["width"] = w;
        images_tmp["id"] = image_id;
        coco_data["images"].append(images_tmp);


        for j in range(len(tmp)//5):
            x1 = int(tmp[j*5+0]);
            y1 = int(tmp[j*5+1]);
            x2 = int(tmp[j*5+2]);
            y2 = int(tmp[j*5+3]);
            label = tmp[j*5+4];
            annotations_tmp = {};
            annotations_tmp["id"] = annotation_id;
            annotation_id += 1;
            annotations_tmp["image_id"] = image_id;
            annotations_tmp["segmentation"] = [];
            annotations_tmp["ignore"] = 0;
            annotations_tmp["area"] = (x2-x1)*(y2-y1);
            annotations_tmp["iscrowd"] = 0;
            annotations_tmp["bbox"] = [x1, y1, x2-x1, y2-y1];
            annotations_tmp["category_id"] = anno.index(label);

            coco_data["annotations"].append(annotations_tmp)
        image_id += 1;
    
    os.system("rm tmp.txt")

    return coco_data;


def preds_to_voc_format(img_name, preds, classes_selected, class_names=[], thresh=0.5):
    if(len(classes_selected) == 1 and classes_selected[0] == ""):
        classes_selected = class_names;
    combined = [];
    wr = "";
    img = cv2.imread(img_name);
    h, w, c = img.shape;
    writer = Writer(img_name, w, h)
    i = 0;
    for j in range(len(preds[i]['rois'])):
        x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
        class_name = obj_list[preds[i]['class_ids'][j]]
        score = float(preds[i]['scores'][j])
        
        if(score >= thresh and class_name in classes_selected):
            writer.addObject(class_name, x1, y1, x2, y2);
            
    return writer


def preds_to_yolo_format(img_name, preds, classes_selected, class_names=[], thresh=0.5):
    if(len(classes_selected) == 1 and classes_selected[0] == ""):
        classes_selected = class_names;
    combined = [];
    wr = "";
    img = cv2.imread(img_name);
    height, width, c = img.shape;
    i = 0;
    for j in range(len(preds[i]['rois'])):
        x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
        class_name = obj_list[preds[i]['class_ids'][j]]
        score = float(preds[i]['scores'][j])
        if(score >= thresh and class_name in classes_selected):
            x_c = str(((x1 + x2)/2)/width);
            y_c = str(((y1 + y2)/2)/height);
            w = str((x2 - x1)/width);
            h = str((y2 - y1)/height);
            class_index = str(classes_selected.index(class_name));
            wr += class_index + " " + x_c + " " + y_c + " " + w + " " + h;
            wr += "\n";
    
    return wr




print("Running Inference");

print("If running for the first time, it might take a little longer ....");

f = open("test_folder.txt");
lines = f.readlines();
f.close();


folder_path = lines[0][:len(lines[0])-1];
data_name = lines[1][:len(lines[1])-1];
model_name = lines[2][:len(lines[2])-1];
output_folder_path = lines[3][:len(lines[3])-1];
thresh = float(lines[4][:len(lines[4])-1]);
classes = lines[5][:len(lines[5])-1];
write_voc_format = int(lines[6][:len(lines[6])-1]);
write_coco_format = int(lines[7][:len(lines[7])-1]);
write_monk_format = int(lines[8][:len(lines[8])-1]);
write_yolo_format = int(lines[9][:len(lines[9])-1]);
visualize = int(lines[10][:len(lines[10])-1]);
savefig = int(lines[11]);


df_status = pd.read_csv(output_folder_path + "/status.csv");

classes = classes.split(",");
if(len(classes) > 0 and classes[0] != ""):
    for i in range(len(classes)):
        classes[i] = classes[i].replace(" ", "");

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

if(len(classes) == 1 and classes[0] == ""):
    classes = obj_list;

if(not os.path.isdir("weights")):
    os.mkdir("weights");

if("0" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d0.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth");
    compound_coef = 0;
elif("1" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d1.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth -O weights/efficientdet-d1.pth");
    compound_coef = 1;
elif("2" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d2.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth -O weights/efficientdet-d2.pth");
    compound_coef = 2;
elif("3" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d3.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth -O weights/efficientdet-d3.pth");
    compound_coef = 3;
elif("4" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d4.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth -O weights/efficientdet-d4.pth");
    compound_coef = 4;
elif("5" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d5.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth -O weights/efficientdet-d5.pth");
    compound_coef = 5;
elif("6" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d6.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth -O weights/efficientdet-d6.pth");
    compound_coef = 6;
elif("7" in model_name):
    print("Downloading model ....");
    if(not os.path.isfile("weights/efficientdet-d7.pth")):
        os.system("wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth -O weights/efficientdet-d7.pth");
    compound_coef = 7;


force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = thresh
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True





color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size



# Load model
print("Loading Model ...");
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model = model.eval()


if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


for k in range(len(df_status)):
    img_name_short = df_status.iloc[k]["id"];
    status = df_status.iloc[k]["status"];

    if(not status):
        print("Processing {}/{}".format(k+1, len(df_status)));
        img_path = folder_path + "/" + img_name_short;
        output_name = output_folder_path + "/" + img_name_short;


        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)



        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)


        preds = invert_affine(framed_metas, out)

        imgs = ori_imgs;
        i = 0;

        if(savefig):
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                if(score >= threshold and obj in classes):
                    plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


            cv2.imwrite(output_name, imgs[i])



        if(write_monk_format):
            print("Saving Annotations to monk format");
            df = preds_to_monk_format(img_path, preds, classes, class_names=obj_list, thresh=thresh);
            out_file_name = output_name.split(".")[0] + ".csv";
            df.to_csv(out_file_name, index=False);


        if(write_coco_format):
            print("Saving Annotations to coco format (individual files)");
            coco_json = preds_to_coco_format(img_path, preds, classes, class_names=obj_list, thresh=thresh);
            out_file_name = output_name.split(".")[0] + ".json";
            outfile =  open(out_file_name, 'w');
            json_str = json.dumps(coco_json, indent=4);
            outfile.write(json_str);
            outfile.close();


        if(write_voc_format):
            print("Saving Annotations to voc format");
            voc_xml = preds_to_voc_format(img_path, preds, classes, class_names=obj_list, thresh=thresh);
            out_file_name = output_name.split(".")[0] + ".xml";
            voc_xml.save(out_file_name)

        if(write_yolo_format):
            print("Saving Annotations to yolo format");
            yolo_str = preds_to_yolo_format(img_path, preds, classes, class_names=obj_list, thresh=thresh);
            out_file_name = output_name.split(".")[0] + ".txt";
            f = open(out_file_name, 'w');
            f.write(yolo_str);
            f.close();

            class_file_name = output_name.split(".")[0].split("/");
            class_file_name = class_file_name[:len(class_file_name)-1];
            class_file_name = "/".join(class_file_name) + "/classes.txt";
            f = open(class_file_name, 'w');
            for i in range(len(classes)):
                f.write(classes[i] + "\n");
            f.close();
        df_status.at[k, "status"] = 1
        df_status.to_csv(output_folder_path + "/status.csv", index=False);
    else:
        print("Already Processed {}/{}".format(k+1, len(df_status)));



print("Completed")