## Project Details
Pipeline based on Pytorch PeleeNet project - https://github.com/yxlijun/Pelee.Pytorch
<br />
<br />
<br />

# Supported Models
  - PeleeNet-304
   

<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0 (Other cuda version support is experimental)
    
`cd installation`

`chmod +x install.sh && ./install.sh`

<br />
<br />
<br />


## Pipeline

- Load Dataset

`gtf.Train_Dataset(train_img_dir, train_label_dir, label_file);`

`gtf.Dataset_Params(batch_size=16, num_workers=3);`

- Load Model

`gtf.Model_Params(gpu_devices=[0]);`

- Set Hyper Parameters

`gtf.Hyper_Params(lr=0.01, gamma=0.1, momentum=0.9, weight_decay=0.0005);`

`gtf.Training_Params(num_epochs=50, model_output_dir="output");`

- Train

`gtf.Train();`



<br />
<br />
<br />

## TODO

- [ ] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
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
