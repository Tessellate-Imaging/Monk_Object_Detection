## Project Details
Pipeline based on Pytorch RetinaFace project - https://github.com/biubug6/Pytorch_Retinaface
<br />
<br />
<br />

# Supported Models
  - mobilenet
  - resnet
   

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

`gtf.Train_Dataset(img_dir, anno_file);`

- Dataset params

`gtf.Dataset_Params(batch_size=32, num_workers=4);`

- Load Model

`gtf.Model_Params(model_type="mobilenet", use_gpu=True, resume_from=None);`

- Set Hyper Parameters

`gtf.Hyper_Parameters(lr=0.0001, momentum=0.9, weight_decay=0.0005, gamma=0.1);`

`gtf.Training_Params(num_epochs=20, output_dir="weights_trained");`

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
