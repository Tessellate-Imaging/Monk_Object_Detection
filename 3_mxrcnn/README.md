## Project Details
Pipeline based on MX-RCNN project - https://github.com/ijkguo/mx-rcnn

<br />
<br />
<br />

## Installation

Supports 
- Python 3.6
- Cuda 9.0 (Other cuda version support is experimental)
    
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

## Pipeline

- Load Dataset

`set_dataset_params(root_dir="../sample_dataset/", coco_dir="kangaroo", imageset="Images");`

- Load Model

`set_model_params(model_name="vgg16");`

- Set Hyper-parameter

`set_hyper_params(gpus="0", lr=0.001, lr_decay_epoch="1", epochs=4, batch_size=1);`
`set_output_params(log_interval=100, save_prefix="model_vgg16");`

- Preprocess dataset

`set_img_preproc_params(img_short_side=600, img_long_side=1000, 
                       mean=(123.68, 116.779, 103.939), std=(1.0, 1.0, 1.0));`
                    
- Initialize 

`initialize_rpn_params();`

- Invoke data loader and network

`roidb = set_dataset();`
`sym = set_network();`

- Train

`train(sym, roidb);`



<br />
<br />
<br />

## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Add Faster-RCNN support
- [ ] Test on Kaggle and Colab 
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
