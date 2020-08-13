## Project Details
Pipeline based on - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md

<br />
<br />
<br />

# Supported Models
  - ssd_mobilenet_v1
  - ssd_mobilenet_v2
  - ssd_mobilenet_v1_ppn
  - ssd_mobilenet_v1_fpn
  - ssd_resnet50_v1_fpn
  - ssd_mobilenet_v1_0.75_depth
  - ssd_mobilenet_v1_quantized
  - ssd_mobilenet_v1_0.75_depth_quantized
  - ssd_mobilenet_v2_quantized
  - ssdlite_mobilenet_v2
  - ssd_inception_v2
  - faster_rcnn_inception_v2
  - faster_rcnn_resnet50
  - faster_rcnn_resnet50_lowproposals
  - rfcn_resnet101
  - faster_rcnn_resnet101
  - faster_rcnn_resnet101_lowproposals
  - faster_rcnn_inception_resnet_v2_atrous
  - faster_rcnn_inception_resnet_v2_atrous_lowproposals
  - faster_rcnn_nas
  - faster_rcnn_nas_lowproposals
  - ssd_mobilenet_v2_mnasfpn
  - ssd_mobilenet_v3_large
  - ssd_mobilenet_v3_small
  
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
 
 `gtf.set_model_params(model_name="ssd_mobilenet_v3_small")`
 
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














