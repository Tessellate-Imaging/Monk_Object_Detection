## Project Details
Pipeline based on Segmentation_Models project - https://github.com/qubvel/segmentation_models


<br />
<br />
<br />

# Supported Models
  - Unet
  - FPN
  - Linknet
  - PSPNet

<br />
<br />


## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`

Check the cuda version using the command

`nvcc -V`

Select the right requirements file and run 

`cat <selected requirements file> | xargs -n 1 -L 1 pip install`

For example for cuda 9.0

`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`



<br />
<br />
<br />


## Functional Docs
[Link]

<br />
<br />
<br />



## Pipeline

 - Load Dataset
 
 `gtf.Train_Dataset(img_dir, mask_dir, classes_dict, classes_to_train);`
 `gtf.Data_Params(batch_size=2, backbone="efficientnetb3");`
 
 - Load Model
 
 `gtf.Model_Params(model="Unet");`
 
 - Train Params
 
 `gtf.Train_Params(lr=0.0001);`
 
 - Setup
 
 `gtf.Setup();`
 
  - Train
  
  `gtf.Train(num_epochs=40);`
  
  
<br />
<br />
<br />


## TODO

- [ ] Add support for Coco-Type Annotated Datasets
- [ ] Add support for VOC-Type Annotated Dataset
- [ ] Test on Kaggle and Colab 
- [ ] Add validation feature & data pipeline
- [ ] Add Optimizer selection feature
- [ ] Add Multi-GPU training
