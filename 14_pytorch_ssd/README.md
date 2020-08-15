## Project Details
Pipeline based on Pytorch SSD project - https://github.com/qfgaohao/pytorch-ssd

<br />
<br />
<br />

# Supported Models
  - Mobilenet-V1 SSD
  - Mobilenet-V2 SSDLite
  - VGG16 SSD
   

<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0 (Other cuda version support is experimental)
    
`cd installation`

`cat requirements.txt | xargs -n 1 -L 1 pip install`

<br />
<br />
<br />


## Pipeline

- Load Dataset

`gtf.set_train_data_params(train_img_dir, train_label_dir, label_file, batch_size=2, balance_data=False, num_workers=4);`

- Load Model

`gtf.set_model_params(net="mb1-ssd", freeze_base_net=False, 
                         freeze_net=False, use_gpu=True, resume=False, mb2_width_mult=1.0);;`

- Set Hyper Parameters

`gtf.set_lr_params(lr=0.001);`

`gtf.set_optimizer_params(momentum=0.09);`

- Train

`gtf.train(num_epochs=5, val_epoch_interval=2, output_folder="models_dir/");`



<br />
<br />
<br />

## TODO

- [x] Add support for Coco-Type Annotated Datasets
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
