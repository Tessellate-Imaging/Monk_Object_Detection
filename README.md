# Monk - A computer vision toolkit for everyone[![Tweet](https://img.shields.io/twitter/url/https/github.com/tterb/hyde.svg?style=social)](http://twitter.com/share?text=Check%20out%20Monk%20Object%20Detection:%20A%20repository%20for%20object%20detection%20pipelines%20in%20computer%20vision&url=https://github.com/Tessellate-Imaging/Monk_Object_Detection&hashtags=MonkAI,OpenSource,Notebooks,DeepLearning,Tutorial,ObjectDetection,Python,AI) [![](http://hits.dwyl.io/Tessellate-Imaging/Monk_Object_Detection.svg)](http://hits.dwyl.io/Tessellate-Imaging/Monk_Object_Detection) ![](https://tokei.rs/b1/github/Tessellate-Imaging/Monk_Object_Detection) ![](https://tokei.rs/b1/github/Tessellate-Imaging/Monk_Object_Detection?category=files) [![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

A repository of complex computer vision algorithms
  - Single line installation
  - Ingest custom datasets for training and inference
  - Low code syntax
  - Easy deployment
  
<br />
<br />
<br />

## Create real-world Object Detection applications 
<table>
  <tr>
    <td>Wheat detection in field</td>
    <td>Detection in underwater imagery</td>
    <td>Trash Detection</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/wheat-detection-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/sea_tutrle_demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/trash.gif" width=320 height=240></td>
  </tr>
  <tr>
    <td>Object detection in bad lighting</td>
    <td>Tiger detection in wild</td>
    <td>Person detection in infrared imagery</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/obj-det-in-bad-light.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/tiger.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/ir-person-det.gif" width=320 height=240></td>
  </tr>
</table>
  

<br />
<br />

## Create real-world Image Segmentation applications 
<table>
  <tr>
    <td>Road Segmentation in satellite imagery</td>
    <td>Ultrasound nerve segmentation</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/satellite-road-segmentation.gif" width=640 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/ultrasound-nerve-image-segmentat.gif" width=320 height=240></td>
  </tr>
</table>




### Documentation
- [LINK](https://li8bot.github.io/monkai/#/home)

<br />
<br />
<br />

## Important Elements

- A) Training Engine
    - Train models on custom dataset witjh low code syntax
    - Pretrained examples on variety of datasets
    - Useful to train your own detector
    
- B) Inference Engine
    - Original pretrained models (from original authors and implementations) for inferencing and analysing
    - Pretrained models on coco, voc, cityscpaes, type datasets
    - Useful to analyse which algoeithm works best for you
    - Useful to generate semi-accurate annotations (coco, pascal-voc, yolo formats) on a new dataset

<br />
<br />
<br />


## Training engine - Pipelines presented as jupyter notebooks - see example_notebooks
(See the licenses for each pipeline and use accordingly)


- A) GluonCV Finetune
    - [Original Implementation](https://gluon-cv.mxnet.io/build/examples_detection/index.html)
    - [Functional Documentation](https://abhi-kumar.github.io/1_gluoncv_finetune_docs/)
        - SSD with Vgg16
        - SSD with Resnet50
        - SSD with Resnet101
        - SSD with MobileNet1.0
        - YoloV3 with Darknet53
        - YoloV3 with MobileNet1.0
    
- B) TorchVision Finetune: Original
    - [Original Implementation](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
    - [Functional Documentation](https://abhi-kumar.github.io/2_pytorch_finetune_docs/)
        - Faster-RCNN with MobileNet2.0
    
- C) MX-RCNN: Original
    - [Original Implementation](https://github.com/ijkguo/mx-rcnn)
    - [Functional Documentation](https://abhi-kumar.github.io/3_mxrcnn_docs/)
        - Faster-RCNN with VGG16
        - Faster-RCNN with Resnet50
        - Faster-RCNN with Resnet101
    
- D) Efficient-Det: Original
    - [Original Implementation](https://github.com/signatrix/efficientdet)
    - [Functional Documentation](https://abhi-kumar.github.io/4_efficientdet_docs/)

- E) Pytorch-Retinanet: Original
    - [Original Implementation](https://github.com/yhenon/pytorch-retinanet)
    - [Functional Documentation](https://abhi-kumar.github.io/5_pytorch_retinanet_docs/)
        - Resnet18
        - Resnet34
        - Resnet50
        - Resnet101
        - Resnet152
    
- F) CornerNet-Lite: Original
    - [Original Implementation](https://github.com/princeton-vl/CornerNet-Lite)
    - [Functional Documentation](https://abhi-kumar.github.io/6_cornernet_lite_docs/)
        - CornerNet-Saccade
        - CornerNet-Squeeze
    
- G) YOLOV3: Original
    - [Original Implementation](https://github.com/ultralytics/yolov3)
    - [Functional Documentation](https://abhi-kumar.github.io/7_yolov3_docs/)
        - yolov3
        - yolov3s
        - yolov3-spp
        - yolov3-spp3
        - yolov3-tiny
        - yolov3-spp-matrix
        - csresnext50-panet-spp

- H) RFBNet:
    - [Original Implementation](https://github.com/ruinmessi/RFBNet)
    - [Functional Documentation](https://abhi-kumar.github.io/8_pytorch_rfbnet_docs/)
        - VGG16
        - E_VGG16
        - MobileNet
    
- I) Segmentation_Models:
    - [Original Implementation](https://github.com/qubvel/segmentation_models)
    - [Functional Documentation]
        - Unet
        - FPN
        - Linknet
        - PSPNet
        
- J) Pytorch_Efficientdet:
    - [Original Implementation](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
    - [Functional Documentation]
        - efficientdet-d0.pth
        - efficientdet-d1.pth
        - efficientdet-d2.pth
        - efficientdet-d3.pth
        - efficientdet-d4.pth
        - efficientdet-d5.pth
        - efficientdet-d6.pth
        - efficientdet-d7.pth
        
- K) Slim-Yolo-V3:
    - [Original Implementation 1](https://github.com/PengyiZhang/SlimYOLOv3)
    - [Original Implementation 2](https://github.com/erikguo/yolov3)
    - [Functional Documentation]
        
      
<br />
<br />
<br />



## Inference Engine
(See the licenses for each pipeline and use accordingly)

- A) GluonCV Finetune
     - [Original Implementation](https://gluon-cv.mxnet.io/build/examples_detection/index.html)
     - Pretrained models on 
          - COCO Dataset
          - Pascal VOC Dataset
     - Models using
          - SSD
          - faster-rcnn
          - Yolo-V3
          - CenterNet
          
 - B) EfficientDet Pytorch
     - [Original Implementation](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
     - Pretrained models on 
          - COCO Dataset
     - Models using efficient network variants
     
 - C) DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution.
    - [Original Implementation](https://github.com/joe-siyuan-qiao/DetectoRS)
    - Pretrained models on 
        - COCO Dataset
    - Models using
        - Resnet-50
        - RexNext-101


<br />
<br />
<br />

## Installation - Inference engine
 - Check - Monk_Object_Detection/inference_engine/




<br />
<br />
<br />

## Installation - Training engine
- A) GluonCV Finetune
    - Check - Monk_Object_Detection/1_gluoncv_finetune/

- B) TorchVision Finetune
    - Check - Monk_Object_Detection/2_pytorch_finetune/
    
- C) MX-RCNN
    - Check - Monk_Object_Detection/3_mxrcnn/
      
- D) Efficient-Det
    - Check - Monk_Object_Detection/4_efficientdet/
    
- E) Pytorch-Retinanet
    - Check - Monk_Object_Detection/5_pytorch_retinanet/
    
- F) CornerNet-Lite
    - Check - Monk_Object_Detection/6_cornernet_lite/
    
- G) YoloV3
    - Check - Monk_Object_Detection/7_yolov3/
    
- H) RFBNet
    - Check - Monk_Object_Detection/8_pytorch_rfbnet
    
- I) Segmentation_Models
    - Check - Monk_Object_Detection/9_segmentation_models
    
- J) Pytorch EfficientDet
    - Check - Monk_Object_Detection/10_pytorch_efficientdet
   
- K) SlimYoloV3
    - Check - Monk_Object_Detection/11_slimyolov3

<br />
<br />
<br />


## Author
Tessellate Imaging - https://www.tessellateimaging.com/
   
Check out Monk AI - (https://github.com/Tessellate-Imaging/monk_v1)
    
    Monk features
        - low-code
        - unified wrapper over major deep learning framework - keras, pytorch, gluoncv
        - syntax invariant wrapper

    Enables developers
        - to create, manage and version control deep learning experiments
        - to compare experiments across training metrics
        - to quickly find best hyper-parameters

To contribute to Monk AI or Monk Object Detection repository raise an issue in the git-repo or dm us on linkedin 
   - Abhishek - https://www.linkedin.com/in/abhishek-kumar-annamraju/
   - Akash - https://www.linkedin.com/in/akashdeepsingh01/
<br />
<br />
<br />


## Copyright

Copyright 2019 onwards, Tessellate Imaging Private Limited Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
