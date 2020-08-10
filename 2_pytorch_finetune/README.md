## Project Details
Pipeline based on TorchVision Fintuning project - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

<br />
<br />
<br />


## Supported Models
  - Faster-RCNN with MobileNet2.0

<br />
<br />

## Installation

Supports 
- Python 3.6
- Python 3.7
    
`cd installation`

Select the right requirements file and run 

`cat <selected requirements file> | xargs -n 1 -L 1 pip install`


`cat requirements.txt | xargs -n 1 -L 1 pip install`


<br />
<br />
<br />


# Functional Documentation
[Link](https://abhi-kumar.github.io/2_pytorch_finetune_docs/)


## Pipeline

- Load Dataset

`gtf.Dataset([train_root_dir, train_img_dir, train_anno_file], batch_size=batch_size);`

- Load Model

`gtf.Model(model_name, use_pretrained=pretrained, use_gpu=gpu);`

- Set Hyper-parameter

`gtf.Set_Learning_Rate(0.001);`

- Train

`gtf.Train(epochs, params_file);`



<br />
<br />
<br />

## TODO

- [x] Add Faster-RCNN support
- [ ] Add YoloV3 support
- [ ] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [ ] Add support for Base Network Changes
- [x] Test on Kaggle and Colab 
- [ ] Add validation feature & data pipeline
- [ ] Add Optimizer selection feature
- [ ] Enable Learning-Rate Scheduler Support
- [ ] Enable Layer Freezing
- [ ] Set Verbosity Levels
- [ ] Add Project management and version control support (Similar to Monk Classification)
- [ ] Add Graph Visualization Support
- [ ] Enable batch proessing at inference
- [ ] Add feature for top-k output visualization
- [ ] Add Multi-GPU training
- [ ] Auto correct missing or corrupt images - Currently skips them
- [ ] Add Experimental Data Analysis Feature
