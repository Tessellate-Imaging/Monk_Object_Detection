## Project Details
Pipeline based on RFBNet project - https://github.com/ruinmessi/RFBNet


<br />
<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`

`chmod +x install.sh`

`./install.sh`


<br />
<br />
<br />


## Functional Docs
[Link](https://abhi-kumar.github.io/8_pytorch_rfbnet_docs/)

<br />
<br />
<br />



## Pipeline

 - Load Dataset
 
 `gtf.Train_Dataset(root_dir, coco_dir, set_dir, batch_size=4,image_size=512, num_workers=3);`
 
 - Load Model
 
 `gtf.Model(model_name="e_vgg", use_gpu=True, ngpu=1);`
 
 - Set Hyper Params
 
 `gtf.Set_HyperParams(lr=0.0001, momentum=0.9, weight_decay=0.0005, gamma=0.1, jaccard_threshold=0.5);`
 
  - Train
  
  `gtf.Train(epochs=10, log_iters=True, output_weights_dir="weights", saved_epoch_interval=10);`
  
  
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
- [ ] Add Multi-GPU training
- [ ] Auto correct missing or corrupt images - Currently skips them
- [ ] Add Experimental Data Analysis Feature
