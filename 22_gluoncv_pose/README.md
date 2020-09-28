## Project Details
Pipeline based on GluonCV Pose Estimation project - https://gluon-cv.mxnet.io/build/examples_pose/dive_deep_simple_pose.html
<br />
<br />
<br />

# Supported Models
  - simple_pose_resnet18_v1b
  - simple_pose_resnet50_v1b
  - simple_pose_resnet101_v1b
  - simple_pose_resnet101_v1d
  - simple_pose_resnet152_v1b
  - simple_pose_resnet152_v1d
  - mobile_pose_resnet18_v1b
  - mobile_pose_resnet50_v1b
  - mobile_pose_mobilenet1.0
  - mobile_pose_mobilenetv2_1.0
  - mobile_pose_mobilenetv3_large
  - mobile_pose_mobilenetv3_small
  - alpha_pose_resnet101_v1b_coco
   

<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
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

`gtf.Train_Dataset("coco");`

`gtf.Dataset_Params(batch_size=2, num_workers=2, num_joints=17, input_size="256,192");`

- Load Model

`gtf.Model_Params(model_name="simple_pose_resnet18_v1b", mode="symbolic", use_pretrained=True, use_pretrained_base=True);`

- Set Hyper Parameters

`gtf.Hyper_Params(lr=0.01, weight_decay=0.0001, lr_decay=0.1);`

`gtf.Training_Params(num_epochs=100, save_frequency=10, output_dir="output", log_iter_interval=100);`

- Train

`gtf.Train();`



<br />
<br />
<br />

## TODO

- [ ] Add support for Coco-Type Annotated Datasets
- [ ] Add support for VOC-Type Annotated Dataset
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
