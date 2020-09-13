## Project Details
Pipeline based on Pytorch TextSnake project - https://github.com/princewang1994/TextSnake.pytorch
<br />
<br />
<br />

# Supported backend Models
  - vgg16
  - resnet50 (in development)
  
  
   

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

`gtf.Train_Dataset(img_folder, anno_folder, annotation_type="mat");`

`gtf.Dataset_Params(batch_size=2, num_workers=2, input_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225));`

- Load Model

`gtf.Model_Params(model_type="vgg", use_pretrained=True, use_gpu=True, use_distributed=False);`

- Set Hyper Parameters

`gtf.Hyper_Params(optimizer="adam", lr=0.0001, weight_decay=0, gamma=0.1, momentum=0.9);`

`gtf.Training_Params(epochs=5, output_dir="trained_weights", experiment_name="exp", save_freq=10, display_freq=50);;`

- Train

`gtf.Train();`



<br />
<br />
<br />

## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for Mat-Type Annotated Dataset
- [x] Add support for Text-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
- [x] Add Optimizer selection feature
- [x] Enable Learning-Rate Scheduler Support
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
