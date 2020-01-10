## Project Details
Pipeline based on EfficientDet project - https://github.com/signatrix/efficientdet

<br />
<br />
<br />

## Installation

Supports 
- Python 3.6
- Cuda 9.0 (Other cuda version support is experimental)
    
`cd installation`

`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`

<br />
<br />
<br />

## Pipeline

- Load Dataset

`
root_dir = "../sample_dataset";
coco_dir = "kangaroo";
img_dir = "images";
set_dir = "Images";
gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=512, use_gpu=True)
`

- Load Model

`gtf.Model();`

- Set Hyper Parameters

`gtf.Set_Hyperparams(lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0)`

- Train

`gtf.Train(num_epochs=2, model_output_dir="trained/");`



<br />
<br />
<br />

## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
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
