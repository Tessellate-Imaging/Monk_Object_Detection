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

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import json
from pascal_voc_writer import Writer

def preds_to_monk_format(img_name, bounding_boxes, class_IDs, scores, class_names=[], thresh=0.5):
    combined = [];
    wr = "";
    for i in range(len(class_IDs[0])):
        class_id = int(class_IDs[0][i][0].asnumpy());
        score = float(scores[0][i][0].asnumpy())
        bbox = bounding_boxes[0][i].asnumpy();
        x1 = int(bbox[0]);
        y1 = int(bbox[1]);
        x2 = int(bbox[2]);
        y2 = int(bbox[3]);
        if(class_id != -1 and score >= thresh):
            class_name = class_names[class_id]
            wr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + class_name + " ";
    wr = wr[:len(wr)-1];
    
    combined.append([img_name, wr]);
    df = pd.DataFrame(combined, columns = ['ID', 'Label'])  
    return df;

def preds_to_coco_format(img_name, bounding_boxes, class_IDs, scores, class_names=[], thresh=0.5):
    combined = [];
    wr = "";
    for i in range(len(class_IDs[0])):
        class_id = int(class_IDs[0][i][0].asnumpy());
        score = float(scores[0][i][0].asnumpy())
        bbox = bounding_boxes[0][i].asnumpy();
        x1 = int(bbox[0]);
        y1 = int(bbox[1]);
        x2 = int(bbox[2]);
        y2 = int(bbox[3]);
        if(class_id != -1 and score >= thresh):
            class_name = class_names[class_id]
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


def preds_to_voc_format(img_name, bounding_boxes, class_IDs, scores, class_names=[], thresh=0.5):
    combined = [];
    wr = "";
    img = cv2.imread(img_name);
    h, w, c = img.shape;
    writer = Writer(img_name, w, h)
    for i in range(len(class_IDs[0])):
        class_id = int(class_IDs[0][i][0].asnumpy());
        score = float(scores[0][i][0].asnumpy())
        bbox = bounding_boxes[0][i].asnumpy();
        x1 = int(bbox[0]);
        y1 = int(bbox[1]);
        x2 = int(bbox[2]);
        y2 = int(bbox[3]);
        if(class_id != -1 and score >= thresh):
            class_name = class_names[class_id]
            writer.addObject(class_name, x1, y1, x2, y2);
            
    return writer


def preds_to_yolo_format(img_name, bounding_boxes, class_IDs, classes_selected, scores, class_names=[], thresh=0.5):
    if(len(classes_selected) == 1 and classes_selected[0] == ""):
        classes_selected = class_names;
    combined = [];
    wr = "";
    img = cv2.imread(img_name);
    height, width, c = img.shape;
    for i in range(len(class_IDs[0])):
        class_id = int(class_IDs[0][i][0].asnumpy());
        score = float(scores[0][i][0].asnumpy())
        bbox = bounding_boxes[0][i].asnumpy();
        x1 = int(bbox[0]);
        y1 = int(bbox[1]);
        x2 = int(bbox[2]);
        y2 = int(bbox[3]);
        if(class_id != -1 and score >= thresh):
            class_name = class_names[class_id]
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


print("Downloading and Loading model")
net = model_zoo.get_model(model_name + "_" + data_name, pretrained=True)


df_status = pd.read_csv(output_folder_path + "/status.csv");
classes = classes.split(",");

for j in range(len(df_status)):
    img_name_short = df_status.iloc[j]["id"];
    status = df_status.iloc[j]["status"];

    if(not status):
        print("Processing {}/{}".format(j+1, len(df_status)));
        img_name = folder_path + "/" + img_name_short;
        output_name = output_folder_path + "/" + img_name_short;
        x, img = data.transforms.presets.center_net.load_test(img_name, short=512)
        class_IDs, scores, bounding_boxes = net(x)
        all_classes = net.classes


        if(len(classes) > 0 and classes[0] != ""):
            for i in range(len(classes)):
                classes[i] = classes[i].replace(" ", "");
            for i in range(len(class_IDs[0])):
                if all_classes[int(class_IDs.asnumpy()[0][i])] not in classes:
                    class_IDs[0][i][0] = -1;


        if(savefig):
            ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                                     class_IDs[0], class_names=net.classes,
                                     thresh=thresh)

            plt.savefig(output_name);


        if(write_monk_format):
            print("Saving Annotations to monk format");
            df = preds_to_monk_format(img_name, bounding_boxes, class_IDs, scores, class_names=net.classes, thresh=0.5);
            out_file_name = output_name.split(".")[0] + ".csv";
            df.to_csv(out_file_name, index=False);


        if(write_coco_format):
            print("Saving Annotations to coco format (individual files)");
            coco_json = preds_to_coco_format(img_name, bounding_boxes, class_IDs, scores, class_names=net.classes, thresh=0.5);
            out_file_name = output_name.split(".")[0] + ".json";
            outfile =  open(out_file_name, 'w');
            json_str = json.dumps(coco_json, indent=4);
            outfile.write(json_str);
            outfile.close();


        if(write_voc_format):
            print("Saving Annotations to voc format");
            voc_xml = preds_to_voc_format(img_name, bounding_boxes, class_IDs, scores, class_names=net.classes, thresh=0.5);
            out_file_name = output_name.split(".")[0] + ".xml";
            voc_xml.save(out_file_name)

        if(write_yolo_format):
            print("Saving Annotations to yolo format");
            yolo_str = preds_to_yolo_format(img_name, bounding_boxes, class_IDs, classes, scores, class_names=net.classes, thresh=0.5);
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
        df_status.at[j, "status"] = 1
        df_status.to_csv(output_folder_path + "/status.csv", index=False);
    else:
        print("Already Processed {}/{}".format(j+1, len(df_status)));



print("Completed");




