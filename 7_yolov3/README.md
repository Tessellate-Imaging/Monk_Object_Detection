## Project Details
Pipeline based on YoloV3 project - https://github.com/ultralytics/yolov3

<br />
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
 
 `gtf.set_train_dataset(img_dir, label_dir, class_list_file, batch_size=2)`
 
 - Load Model
 
 `gtf.set_model(model_name="yolov3");`
 
 - Set Hyper Params
 
 `gtf.set_hyperparams(optimizer="sgd", lr=0.00579, multi_scale=False, evolve=True, num_generations=2);`
 
  - Train
  
  `gtf.Train(num_epochs=2);`
  
  
<br />
<br />
<br />

## TODO

- [ ] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
- [ ] Resolve Error with original cornernet model
- [x] Add Optimizer selection feature
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


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
