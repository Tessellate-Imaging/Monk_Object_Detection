## Project Details
Pipeline based on GluonCV Fintuning project - https://gluon-cv.mxnet.io/build/examples_detection/index.html

<br />
<br />
<br />

## Installation

Supports 
- Python 3.6
- Python 3.7
    
`cd installation`

Check the cuda version using the command

`nvcc -V`

Select the right requirements file and run 

`cat <selected requirements file> | xargs -n 1 -L 1 pip install`

For example for cuda 9.0

`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`


<br />
<br />
<br />

## Functional Documentation
  [Link](https://abhi-kumar.github.io/1_gluoncv_finetune_docs/)

<br />
<br />
<br />

## Pipeline

- Load Dataset

`gtf.Dataset(root_dir, img_dir, anno_file, batch_size=batch_size);`

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

- [x] Add SSD support
- [x] Add YoloV3 support
- [ ] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [ ] Add Faster-RCNN support
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
- [x] Add Multi-GPU training
- [ ] Auto correct missing or corrupt images - Currently skips them
- [ ] Add Experimental Data Analysis Feature


<br />
<br />
<br />

## External Contributors list 

- https://github.com/THEFASHIONGEEK: Multi GPU feature
