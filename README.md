# [Monk - A computer vision toolkit for everyone](https://li8bot.github.io/monkai/#/home) [![Tweet](https://img.shields.io/twitter/url/https/github.com/tterb/hyde.svg?style=social)](http://twitter.com/share?text=Check%20out%20Monk%20Object%20Detection:%20A%20repository%20for%20object%20detection%20pipelines%20in%20computer%20vision&url=https://github.com/Tessellate-Imaging/Monk_Object_Detection&hashtags=MonkAI,OpenSource,Notebooks,DeepLearning,Tutorial,ObjectDetection,Python,AI) [![](http://hits.dwyl.io/Tessellate-Imaging/Monk_Object_Detection.svg)](http://hits.dwyl.io/Tessellate-Imaging/Monk_Object_Detection) ![](https://tokei.rs/b1/github/Tessellate-Imaging/Monk_Object_Detection) ![](https://tokei.rs/b1/github/Tessellate-Imaging/Monk_Object_Detection?category=files) [![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

### Monk Object Detection - A low code wrapper over state-of-the-art deep learning algorithms

<br />

### Why use Monk
 - Issue: Abudance of algorithms and difficult to find a working code
   - <b> Solution: All your state-of-the-art as well as old algorithms in one place </b>
 
 - Issue: Installaing different deep learning pipelines is an error-prone task
   - <b> Solution: Single line installations with monk </b>
 
 - Issue: Setting up different algorithms for your custom data requires a lot of effort in changing the existing codes
   - <b> Solution: Easily ingest your custom data for training in COCO, VOC, or Yolo formats </b>
 
 - Issue: Difficulty to trace out which hyperparameters to change for tuning the algorithm
   - <b> Solution: Set your hyper-parameters with a common structure for every algorithm </b>
 
 - Issue: Deployment requires knowledge of base libraries and codes
   - <b> Solution: Easily deploy your models using Monk's low code-syntax </b>
   
 - Issue: Looking for hands-on tutorials for computer vision
   - <b> Solution: Use monk's application building tutorial set</b>
   
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
  
### For more such tutorials visit [Application Model Zoo](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/application_model_zoo)  
  

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

### For more such tutorials visit [Application Model Zoo](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/application_model_zoo)

<br />
<br />

## Other applications 
<table>
  <tr>
    <td>Face Detection</td>
    <td>Pose Estimation</td>
    <td>Activity Recognition</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/face.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/coming_soon.jpg" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/ucf-demo.gif" width=320 height=240></td>
  </tr>
  <tr>
    <td>Object Re-identification</td>
    <td>Object Tracking</td>
    <td>Scene Text Localization</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/coming_soon.jpg" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/coming_soon.jpg" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_det_demos/blob/master/text_demo.gif" width=320 height=240></td>
  </tr>
</table>

### For more such tutorials visit [Application Model Zoo](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/application_model_zoo)

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



## Training Engine Algorithms
    - Train models on custom dataset witjh low code syntax
    - Pretrained examples on variety of datasets
    - Useful to train your own detector

| S.No. | Algorithm Type     | Algorithm                       | Model variations | Installation                                                                                            | Example Notebooks                                                                                                         | Code                                                                                                    | Credits                                                                                         | Functional Docs                                                |
|-------|--------------------|---------------------------------|------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| 1     | Object Detection   | GluonCV Finetune                | 5                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/1_gluoncv_finetune)      | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/1_gluoncv_finetune)      | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/1_gluoncv_finetune)      | [LINK](https://gluon-cv.mxnet.io/build/examples_detection/index.html)                           | [LINK](https://abhi-kumar.github.io/1_gluoncv_finetune_docs/)  |
| 2     | Object Detection   | Tensorflow Object Detection 1.0 | 22               | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/12_tf_obj_1)             | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/12_tf_obj_1)             | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/12_tf_obj_1)             | [LINK](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md) | In Development                                                 |
| 3     | Object Detection   | Tensorflow Object Detection 2.0 | 26               | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/13_tf_obj_2)             | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/13_tf_obj_2)             | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/13_tf_obj_2)             | [LINK](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) | In Development                                                 |
| 4     | Object Detection   | Pytorch Efficient-Det 1         | 1                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/4_efficientdet)          | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/4_efficientdet)          | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/4_efficientdet)          | [LINK](https://github.com/signatrix/efficientdet)                                               | [LINK](https://abhi-kumar.github.io/4_efficientdet_docs/)      |
| 5     | Object Detection   | Pytorch Efficient-Det 2         | 8                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/10_pytorch_efficientdet) | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/10_pytorch_efficientdet) | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/10_pytorch_efficientdet) | [LINK](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                             | In Development                                                 |
| 6     | Object Detection   | TorchVision Finetune            | 1                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/2_pytorch_finetune)      | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/2_pytorch_finetune)      | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/2_pytorch_finetune)      | [LINK](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)                    | [LINK](https://abhi-kumar.github.io/2_pytorch_finetune_docs/)  |
| 7     | Object Detection   | Mx-RCNN                         | 3                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/3_mxrcnn)                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/3_mxrcnn)                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/3_mxrcnn)                | [LINK](https://github.com/ijkguo/mx-rcnn)                                                       | [LINK](https://abhi-kumar.github.io/3_mxrcnn_docs/)            |
| 8     | Object Detection   | Pytorch-Retinanet               | 5                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/5_pytorch_retinanet)     | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/5_pytorch_retinanet)     | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/5_pytorch_retinanet)     | [LINK](https://github.com/yhenon/pytorch-retinanet)                                             | [LINK](https://abhi-kumar.github.io/5_pytorch_retinanet_docs/) |
| 9     | Object Detection   | CornerNet Lite                  | 2                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/6_cornernet_lite)        | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/6_cornernet_lite)        | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/6_cornernet_lite)        | [LINK](https://github.com/princeton-vl/CornerNet-Lite)                                          | [LINK](https://abhi-kumar.github.io/6_cornernet_lite_docs/)    |
| 10    | Object Detection   | YoloV3                          | 7                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/7_yolov3)                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/7_yolov3)                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/7_yolov3)                | [LINK](https://github.com/ultralytics/yolov3)                                                   | [LINK](https://abhi-kumar.github.io/7_yolov3_docs/)            |
| 11    | Object Detection   | RFBNet                          | 3                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/8_pytorch_rfbnet)        | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/8_pytorch_rfbnet)        | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/8_pytorch_rfbnet)        | [LINK](https://github.com/ruinmessi/RFBNet)                                                     | [LINK](https://abhi-kumar.github.io/8_pytorch_rfbnet_docs/)    |
| 12    | Object Detection   | Slim-Yolo-V3                    | 1                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/11_slimyolov3)           | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/11_slimyolov3)           | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/11_slimyolov3)           | [LINK](https://github.com/PengyiZhang/SlimYOLOv3)                                               | In Development                                                 |
| 13    | Object Detection   | Pytorch SSD                     | 3                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/14_pytorch_ssd)          | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/14_pytorch_ssd)          | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/14_pytorch_ssd)          | [LINK](https://github.com/qfgaohao/pytorch-ssd)                                                 | In Development                                                 |
| 14    | Object Detection   | Pytorch-Peleenet                | 1                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/15_pytorch_peleenet)     | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/15_pytorch_peleenet)     | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/15_pytorch_peleenet)     | [LINK](https://github.com/yxlijun/Pelee.Pytorch)                                                | In Development                                                 |
| 15    | Object Detection   | MM-Detection                    | 36               | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/16_mmdet)                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/16_mmdet)                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/16_mmdet)                | [LINK](https://github.com/open-mmlab/mmdetection)                                               | In Development                                                 |
| 16    | Image Segmentation | Segmentation Models             | 4                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/9_segmentation_models)   | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/9_segmentation_models)   | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/9_segmentation_models)   | [LINK](https://github.com/qubvel/segmentation_models)                                           | In Development                                                 |
| 17    | Pytorch Retinaface | Face Detection                  | 2                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/17_retinaface)           | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/17_retinaface)           | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/17_retinaface)           | [LINK](https://github.com/biubug6/Pytorch_Retinaface)                                           | In Development                                                 |
| 18    | Action Recognition | MM-Action2                      | 1                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/1_gluoncv_finetune)      | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/18_mmaction)             | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/18_mmaction)             | [LINK](https://github.com/open-mmlab/mmaction2)                                                 | In Development                                                 |
| 19    | Text Localization  | Pytorch-TextSnake               | 1                | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/19_pytorch_textsnake)    | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/example_notebooks/19_pytorch_textsnake)    | [LINK](https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/19_pytorch_textsnake)    | [LINK](https://github.com/princewang1994/TextSnake.pytorch)                                     | In Development                                                 |

<br />
<br />
<br />


## Aknowledgements
  - Contributors' information can be found here: https://github.com/Tessellate-Imaging/Monk_Object_Detection/blob/master/Contributors.md
  - Majority of the code is obtained from these pipelines (Monk is a low code wrapper to utilize these pipelines)
    - https://gluon-cv.mxnet.io/build/examples_detection/index.html
    - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md
    - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
    - https://github.com/signatrix/efficientdet
    - https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
    - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    - https://github.com/ijkguo/mx-rcnn
    - https://github.com/yhenon/pytorch-retinanet
    - https://github.com/princeton-vl/CornerNet-Lite
    - https://github.com/ultralytics/yolov3
    - https://github.com/ruinmessi/RFBNet
    - https://github.com/PengyiZhang/SlimYOLOv3
    - https://github.com/qfgaohao/pytorch-ssd
    - https://github.com/open-mmlab/mmdetection
    - https://github.com/qubvel/segmentation_models
    - https://github.com/biubug6/Pytorch_Retinaface
    - https://github.com/open-mmlab/mmaction2


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
