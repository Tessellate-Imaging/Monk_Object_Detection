# EfficientDet: Scalable and Efficient Object Detection

## Introduction

Here is our pytorch implementation of the model described in the paper **EfficientDet: Scalable and Efficient Object Detection** [paper](https://arxiv.org/abs/1911.09070) (*Note*: We also provide pre-trained weights, which you could see at ./trained_models) 
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>An example of our model's output.</i>
</p>


## Datasets


| Dataset                | Classes |    #Train images      |    #Validation images      |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2017               |    80   |          118k         |              5k            |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:
  ```
  COCO
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  │── images
      ├── train2017
      └── val2017
  ```
  
## How to use our code

With our code, you can:

* **Train your model** by running **python train.py**
* **Evaluate mAP for COCO dataset** by running **python mAP_evaluation.py**
* **Test your model for COCO dataset** by running **python test_dataset.py --pretrained_model path/to/trained_model**
* **Test your model for video** by running **python test_video.py --pretrained_model path/to/trained_model --input path/to/input/file --output path/to/output/file**

## Experiments

We trained our model by using 3 NVIDIA GTX 1080Ti. Below is mAP (mean average precision) for COCO val2017 dataset 

|   Average Precision   |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.314   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |      IoU=0.50     |   area=   all   |   maxDets=100   |   0.461   |
|   Average Precision   |      IoU=0.75     |   area=   all   |   maxDets=100   |   0.343   |
|   Average Precision   |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.093   |
|   Average Precision   |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.358   |
|   Average Precision   |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.517   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=1     |   0.268   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=10    |   0.382   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.403   |
|     Average Recall    |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.117   |
|     Average Recall    |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.486   |
|     Average Recall    |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.625   |


## Results

Some predictions are shown below:

<img src="demo/1.jpg" width="280"> <img src="demo/2.jpg" width="280"> <img src="demo/3.jpg" width="280">

<img src="demo/4.jpg" width="280"> <img src="demo/5.jpg" width="280"> <img src="demo/6.jpg" width="280">

<img src="demo/7.jpg" width="280"> <img src="demo/8.jpg" width="280"> <img src="demo/9.jpg" width="280">


## Requirements

* **python 3.6**
* **pytorch 1.2**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **pycocotools**
* **efficientnet_pytorch**

## References
- Mingxing Tan, Ruoming Pang, Quoc V. Le. "EfficientDet: Scalable and Efficient Object Detection." [EfficientDet](https://arxiv.org/abs/1911.09070).
- Our implementation borrows some parts from [RetinaNet.Pytorch](https://github.com/yhenon/pytorch-retinanet)
  

## Citation

    @article{EfficientDetSignatrix,
        Author = {Signatrix GmbH},
        Title = {A Pytorch Implementation of EfficientDet Object Detection},
        Journal = {https://github.com/signatrix/efficientdet},
        Year = {2020}
    }