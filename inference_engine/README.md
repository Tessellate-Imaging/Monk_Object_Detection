## Installation

Supports 
- Python 3.6
- Cuda 9.0, Cuda 10.0

` $ sudo apt-get install python3.6 python3.6-dev python3-pip`

` $ sudo pip install virtualenv virtualenvwrapper`

` $ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc`

` $ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc`

` $ echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc`

` $ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc`

` $ source ~/.bashrc`

 #### Reboot system here and cd back to Monk_Object_Detection/inference_engine

` $ cd installation`

` $ cat requirements.txt | xargs -n 1 -L 1 pip install`


<br />
<br />
<br />

## Project Credits
- A) Object Detection
    - [GluonCV Finetune Model Zoo](https://gluon-cv.mxnet.io/model_zoo/detection.html)
    - [EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
    - [DetectoRS](https://github.com/joe-siyuan-qiao/DetectoRS)

- B) Image Segmentation
    - In Development

- C) Pose Estimation
    - In Development

- D) Face Recognition
    - In Development

- E) Facial Keypoint Recognition
    - In Development

- F) Activity Classification
    - In Development

- G) Gaze Estimation
    - In Development

- H) Object Tracking
    - In Development


<br />
<br />
<br />

## Process

### Import Inference Engine

```
import sys
sys.path.append("Monk_Object_Detection/inference_engine/")

from engine import Infer
gtf = Infer();
```


### List all algorithm types

```
gtf.List_Algo_Types();
```


### List all algorithms in a set type
```
gtf.List_Algos(algo_type="object_detection");
```


### Install An algorithm
```
gtf.Install_Engine(algo_type="object_detection", algo="efficientdet_pytorch", system="cuda-9.0")
```



### List all available pretrained models in an algorithm
```
gtf.List_Model_Names(algo_type="object_detection", algo="efficientdet_pytorch");
```


###  Infer on a single image

```
gtf.Infer_Image(algo_type="object_detection", 
                algo="efficientdet_pytorch",
                data="coco",
                model="efficientdet-d1",
                img_path="test.png",
                thresh=0.3,
                classes = "car, person",
                visualize=True,
                write_voc_format=True,
                write_coco_format=True,
                write_monk_format=True,
                write_yolo_format=True,
                verbose=1);
 ```
 
 
 
###  Infer on a folder of images

```
gtf.Infer_Folder_Images(algo_type="object_detection", 
                        algo="efficientdet_pytorch",
                        data="coco",
                        model="efficientdet-d1",
                        folder_path="data",
                        output_folder_name="data",
                        thresh=0.5,
                        classes = "car, person",
                        visualize=True,
                        write_voc_format=True,
                        write_coco_format=True,
                        write_monk_format=True,
                        write_yolo_format=True,
                        verbose=1);
 ```
 
 
 
### Infer on a Video


```
gtf.Infer_Video(algo_type="object_detection", 
                 algo="efficientdet_pytorch",
                 data="coco",
                 model="efficientdet-d1",
                 video_path="day_bicycle.mp4",
                 fps=1,
                 merge_video=True,
                 output_folder_name="day_bicycle",
                 thresh=0.5,
                 classes = "car, person",
                 write_voc_format=True,
                 write_coco_format=True,
                 write_monk_format=True,
                 write_yolo_format=True,
                 verbose=1);
 ```
           
           

### Run multiple algorithms on a single image and compare

```
# Algo_Name, Data_Trained_On, Model_Name
model_list=[["gluoncv_finetune", "coco", "center_net_resnet101_v1b"],
            ["gluoncv_finetune", "coco", "ssd_512_resnet50_v1"],
            ["gluoncv_finetune", "coco", "yolo3_darknet53"],
            ["efficientdet_pytorch", "coco", "efficientdet-d1"],
            ["efficientdet_pytorch", "coco", "efficientdet-d7"]
           ]

gtf.Compare_On_Image(algo_type="object_detection", 
                    model_list=model_list,
                    img_path="5.png",
                    thresh=0.5,
                    classes = classes1,
                    visualize=True,
                    write_voc_format=True,
                    write_coco_format=True,
                    write_monk_format=True,
                    write_yolo_format=True,
                    verbose=1);
                    
```



           
