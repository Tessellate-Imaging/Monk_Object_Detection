## Project Details
Pipeline based on SlimYoloV3 project - https://github.com/PengyiZhang/SlimYOLOv3
Base Yolov3 code from - https://github.com/erikguo/yolov3

<br />
<br />
<br />

# Supported Models
  - YoloV3-SPP3

<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`

`cat requirements.txt | xargs -n 1 -L 1 pip install`


<br />
<br />
<br />


## Pipeline

 - Load Dataset
 
 `gtf.set_train_dataset(img_dir, label_dir, class_list_file, batch_size=2, img_size=608)`
 
 - Set Hyper Params
 
 `gtf.set_hyperparams(optimizer="sgd", lr=0.00579, multi_scale=False, evolve=True, num_generations=2);`
 
  - Train
  
  `gtf.Train(num_epochs=2);`
 
  - Reload Model, prune and retrain
  
  `gtf.prune_weights("yolov3-spp3.cfg", "weights/last.pt", "pruned1.cfg", "pruned1.pt"1);`
    
  `gtf.Train(num_epochs=2, finetune=True)`

<br />
<br />
<br />


## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
- [x] Add Optimizer selection feature
- [x] Enable Learning-Rate Scheduler Support
- [x] Enable Layer Freezing
- [ ] Set Verbosity Levels
- [ ] Add Project management and version control support (Similar to Monk Classification)
- [ ] Add Graph Visualization Support
- [ ] Enable batch proessing at inference
- [ ] Add feature for top-k output visualization
- [ ] Add Multi-GPU training
- [ ] Auto correct missing or corrupt images - Currently skips them
- [ ] Add Experimental Data Analysis Feature
