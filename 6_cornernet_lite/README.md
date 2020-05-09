## Project Details
Pipeline based on CornerNet-Lite project - https://github.com/princeton-vl/CornerNet-Lite

<br />
<br />
<br />

## Installation

Supports 
- Python 3.7
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`

`chmod +x install.sh`

`./install.sh`


<br />
<br />
<br />

## Functional Documentation
[Link](https://abhi-kumar.github.io/6_cornernet_lite_docs/)



## Pipeline

- Load Dataset

`gtf.Train_Dataset(root_dir="../sample_dataset", coco_dir="kangaroo", img_dir="images", set_dir="Train", batch_size=8, image_size=512, use_gpu=True)`

 - Select Model Type
 
 `gtf.Model(model_name="CornerNet_Saccade")`
 
 - Set Hyper-parameters
 
 `gtf.Hyper_Params(lr=0.00025, total_iterations=1000)`
 
 - Setup system
 
 `gtf.Setup()`
 
 - Train
 
 `gtf.Train()`
 
 
 <br />
<br />
<br />

## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
- [ ] Resolve Error with original cornernet model
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

 
 
 


