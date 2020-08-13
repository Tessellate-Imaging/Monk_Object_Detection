## Project Details
Pipeline based on - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

<br />
<br />
<br />

# Supported Models
  - ssd_mobilenet_v2_320
  - ssd_mobilenet_v1_fpn_640
  - ssd_mobilenet_v2_fpnlite_320
  - ssd_mobilenet_v2_fpnlite_640
  - ssd_resnet50_v1_fpn_320
  - ssd_resnet50_v1_fpn_640
  - ssd_resnet101_v1_fpn_320
  - ssd_resnet101_v1_fpn_640
  - ssd_resnet152_v1_fpn_320
  - ssd_resnet152_v1_fpn_640
  - faster_rcnn_resnet50_v1_640
  - faster_rcnn_resnet50_v1_1024
  - faster_rcnn_resnet101_v1_640
  - faster_rcnn_resnet101_v1_1024
  - faster_rcnn_resnet152_v1_640
  - faster_rcnn_resnet152_v1_1024
  - faster_rcnn_inception_resnet_v2_640
  - faster_rcnn_inception_resnet_v2_1024
  - efficientdet_d0
  - efficientdet_d1
  - efficientdet_d2
  - efficientdet_d3
  - efficientdet_d4
  - efficientdet_d5
  - efficientdet_d6
  - efficientdet_d7
  
<br />
<br />

## Installation

Supports 
- Python 3.6
- Cuda 10.0 (Other cuda version support is experimental)
- Tensorflow 1.15
    
`cd installation`

`chmod +x install_cuda10.sh && ./install_cuda10.sh`


<br />
<br />
<br />


## Pipeline

 - Load Dataset
 
 `gtf.set_train_dataset(train_img_dir, train_anno_dir, class_list_file, batch_size=2, trainval_split = 0.8)`
 
 `gtf.create_tfrecord(data_output_dir="data_tfrecord")`
 
 - Set Model
 
 `gtf.set_model_params(model_name="ssd_mobilenet_v2_320")`
 
 - Set Hyper Params
 
 `gtf.set_hyper_params(num_train_steps=1000, lr=0.1)`
  
  - Train
  
  `%run Monk_Object_Detection/12_tf_obj_1/lib/train.py`
 
  - Export Model
  
  `%run Monk_Object_Detection/12_tf_obj_1/lib/export.py`



<br />
<br />
<br />


## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
- [x] Add Optimizer selection feature
- [ ] Enable Learning-Rate Scheduler Support
- [x] Enable Layer Freezing
- [ ] Set Verbosity Levels
- [ ] Add Project management and version control support (Similar to Monk Classification)
- [ ] Add Graph Visualization Support
- [ ] Enable batch proessing at inference
- [ ] Add feature for top-k output visualization
- [ ] Add Multi-GPU training
- [ ] Auto correct missing or corrupt images - Currently skips them
- [ ] Add Experimental Data Analysis Feature














