from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random
import json

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm


def dict_to_tf(image_folder,
               data,
               label_map_dict,
               ignore_difficult_instances=False):
  
    full_path = os.path.join(image_folder, data['filename'])
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
  
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    
    if 'object' in data:
        for obj in data['object']:
            if("difficult" in obj.keys()):
                difficult = bool(int(obj['difficult']))
            else:
                difficult = 0;
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            if("truncated" in obj.keys()):
                truncated.append(int(obj['truncated']))
            else:
                truncated.append(0)
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def list_to_pbtxt(class_list):
    f = open("labelmap.txt", 'w');
    num_classes = 0;
    for i in range(len(class_list)):
        class_list[i] = class_list[i][:len(class_list[i])-1]
        if(class_list[i] != ""):
            f.write('item\n')
            f.write('{\n')
            f.write('    id :{}'.format(int(i)+1))
            f.write('\n')
            f.write("    name :'{0}'".format(str(class_list[i])))
            f.write('\n')
            f.write('}\n')
            num_classes += 1;
    f.close();
    return "labelmap.txt", num_classes;


def create():
    with open('system_dict.json') as json_file:
        args = json.load(json_file)
        
    f = open(args["class_list_file"], 'r');
    lines = f.readlines();
    f.close();
             
    
    args["label_map"], args["num_classes"] = list_to_pbtxt(lines);
    
    
    
    if(not os.path.isdir(args["output_path"])):
        os.mkdir(args["output_path"]);
        
    
    label_map_dict = label_map_util.get_label_map_dict(args["label_map"])


    if(not args["only_eval"]):
    
        if(not args["val_anno_dir"]):
            anno_list = os.listdir(args["train_anno_dir"]);
            num_train = int(len(anno_list)*args["trainval_split"])
            random.shuffle(anno_list)
            train_list = anno_list[:num_train];
            val_list = anno_list[num_train:];
            args["val_img_dir"] = args["train_img_dir"]
            args["val_anno_dir"] = args["train_anno_dir"]
        else:
            train_list = os.listdir(args["train_anno_dir"]);
            val_list = os.listdir(args["val_anno_dir"]);
            
        
        if(not os.path.isfile(args["output_path"] + "/train.record")):
            writer = tf.python_io.TFRecordWriter(args["output_path"] + "/train.record")
            print('Reading training dataset.')
            for i in tqdm(range(len(train_list))):

                path = args["train_anno_dir"] + "/" + train_list[i]
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_record = dict_to_tf(args["train_img_dir"],
                                                   data,
                                                   label_map_dict)
                writer.write(tf_record.SerializeToString())

                #break;
            writer.close();
        else:
            print('Training tfrecord already present at {}.'.format(args["output_path"] + "/train.record"))
        
        if(not os.path.isfile(args["output_path"] + "/val.record")):
            writer = tf.python_io.TFRecordWriter(args["output_path"] + "/val.record")
            print('Reading validation dataset.')
            for i in tqdm(range(len(val_list))):
                
                path = args["val_anno_dir"] + "/" + val_list[i]
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_record = dict_to_tf(args["val_img_dir"],
                                                   data,
                                                   label_map_dict)
                writer.write(tf_record.SerializeToString())

                #break;
            writer.close();
        else:
            print('Validation tfrecord already present at {}.'.format(args["output_path"] + "/val.record"))
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(args, json_file)  
    else:
        if(not os.path.isfile(args["output_path"] + "/external_val.record")):
            writer = tf.python_io.TFRecordWriter(args["output_path"] + "/external_val.record")
            print('Reading validation dataset.')
            for i in tqdm(range(len(val_list))):
                
                path = args["val_anno_dir"] + "/" + val_list[i]
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_record = dict_to_tf(args["val_img_dir"],
                                                   data,
                                                   label_map_dict)
                writer.write(tf_record.SerializeToString())

                #break;
            writer.close();
        else:
            print('Validation tfrecord already present at {}.'.format(args["output_path"] + "/val.record"))
        
        with open('system_dict.json', 'w') as json_file:
            json.dump(args, json_file)
    

def create_val():
    with open('system_dict_val.json') as json_file:
        args = json.load(json_file)
        
    f = open(args["class_list_file"], 'r');
    lines = f.readlines();
    f.close();
             
    
    args["label_map"], args["num_classes"] = list_to_pbtxt(lines);
    
    
    
    if(not os.path.isdir(args["output_path"])):
        os.mkdir(args["output_path"]);
        
    
    label_map_dict = label_map_util.get_label_map_dict(args["label_map"])


    if(not args["only_eval"]):
    
        if(not args["val_anno_dir"]):
            anno_list = os.listdir(args["train_anno_dir"]);
            num_train = int(len(anno_list)*args["trainval_split"])
            random.shuffle(anno_list)
            train_list = anno_list[:num_train];
            val_list = anno_list[num_train:];
            args["val_img_dir"] = args["train_img_dir"]
            args["val_anno_dir"] = args["train_anno_dir"]
        else:
            train_list = os.listdir(args["train_anno_dir"]);
            val_list = os.listdir(args["val_anno_dir"]);
            
        
        if(not os.path.isfile(args["output_path"] + "/train.record")):
            writer = tf.python_io.TFRecordWriter(args["output_path"] + "/train.record")
            print('Reading training dataset.')
            for i in tqdm(range(len(train_list))):

                path = args["train_anno_dir"] + "/" + train_list[i]
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_record = dict_to_tf(args["train_img_dir"],
                                                   data,
                                                   label_map_dict)
                writer.write(tf_record.SerializeToString())

                #break;
            writer.close();
        else:
            print('Training tfrecord already present at {}.'.format(args["output_path"] + "/train.record"))
        
        if(not os.path.isfile(args["output_path"] + "/val.record")):
            writer = tf.python_io.TFRecordWriter(args["output_path"] + "/val.record")
            print('Reading validation dataset.')
            for i in tqdm(range(len(val_list))):
                
                path = args["val_anno_dir"] + "/" + val_list[i]
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_record = dict_to_tf(args["val_img_dir"],
                                                   data,
                                                   label_map_dict)
                writer.write(tf_record.SerializeToString())

                #break;
            writer.close();
        else:
            print('Validation tfrecord already present at {}.'.format(args["output_path"] + "/val.record"))
        
        with open('system_dict_val.json', 'w') as json_file:
            json.dump(args, json_file)  
    else:
        val_list = os.listdir(args["val_anno_dir"]);
        if(not os.path.isfile(args["output_path"] + "/external_val.record")):
            writer = tf.python_io.TFRecordWriter(args["output_path"] + "/external_val.record")
            print('Reading validation dataset.')
            for i in tqdm(range(len(val_list))):
                
                path = args["val_anno_dir"] + "/" + val_list[i]
                #print(path)
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_record = dict_to_tf(args["val_img_dir"],
                                                   data,
                                                   label_map_dict)
                writer.write(tf_record.SerializeToString())

                #break;
            writer.close();
        else:
            print('Validation tfrecord already present at {}.'.format(args["output_path"] + "/val.record"))
        
        with open('system_dict_val.json', 'w') as json_file:
            json.dump(args, json_file)
    