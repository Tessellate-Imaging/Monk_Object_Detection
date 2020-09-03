## Project Details
Pipeline based on Open-MMLAB MM-Action-2 project - https://github.com/open-mmlab/mmaction2
<br />
<br />
<br />

# Supported Models
  - tsn50 (Temporal segment networks)
   

<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`

`chmod +x install.sh && ./install.sh`

<br />
<br />
<br />


## Pipeline

- Load Dataset

`gtf.Train_Video_Dataset(video_dir, anno_file,  classes_list_file);`

`gtf.Dataset_Params(batch_size=2, num_workers=2);`

- Load Model

`gtf.Model_Params(model_name="tsn_r50", gpu_devices=[0]);`

- Set Hyper Parameters

`gtf.Hyper_Params(lr=0.02, momentum=0.9, weight_decay=0.0001);`

`gtf.Training_Params(num_epochs=2, val_interval=1);`

- Train

`gtf.Train();`



<br />
<br />
<br />

## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
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
