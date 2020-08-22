### Pelee: A Real-Time Object Detection System on Mobile Devices, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882) The official and original Caffe code can be found [here](https://github.com/Robert-JunWang/Pelee).

### Description  
I train Pelee with pytorch and the result is better than the original paper result,the pretrained model can be downloaded in [peleenet.pth](https://drive.google.com/open?id=1hxQz7NO-cf-Pa9rg5A-G1ruwpSDOyu7a).

### MAP in VOC2007

| Method | 07+12 | 07+12+coco 
|:-------|:-----:|:-------:|
| SSD300 | 77.2 | 81.2|
| SSD+MobileNet | 68 | 72.7|
| Original Pelee | 70.9| 76.4|
| Ours Pelee | [71.76](https://drive.google.com/open?id=16HparGAVhxTDByi5RylYCkxLZYducK9j) |  ---  |

### Preparation
**the supported version is pytorch-0.4.1 or pytorch-1.0**  
* tqdm
* opencv
* addict
* pytorch>=0.4

- Clone this repository.
```Shell
git clone https://github.com/yxlijun/Pelee.Pytorch
```
- Compile the nms and coco tools:

```Shell
sh make.sh
```

- Prepare dataset (e.g., VOC, COCO), refer to [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) for detailed instructions.
### train
you can train different set according to configs/*,First, you should download the pretrained model [peleenet.pth](https://drive.google.com/open?id=1hxQz7NO-cf-Pa9rg5A-G1ruwpSDOyu7a),then,move the file to weights/
```
python train.py --dataset VOC\COCO --config ./configs/Pelee_VOC.py  
```  
if you train with multi gpu    
```  
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset VOC\COCO --config ./configs/Pelee_VOC.py   --ngpu 2
```
### eval
you can evaluate your model in  voc and coco  
```
python test.py --dataset VOC\COCO  --config ./configs/Pelee_VOC.py --trained_model ./weights/Pelee_VOC.pth 
```
### demo 
you can test your image, First, download the trained model [Pelee_VOC.pth](https://drive.google.com/open?id=16HparGAVhxTDByi5RylYCkxLZYducK9j) file. Then, move the file to weights/.
```
python demo.py --dataset VOC\COCO  --config ./configs/Pelee_VOC.py --trained_model ./weights/Pelee_VOC.pth --show  
```
You can see the image with drawed boxes as:
<div align=center><img src="imgs/VOC/im_res/street_stdngit.jpg" width="450" hegiht="163" align=center />

<div align=left>

### TODO 
the code support: 

  * Support for the MS COCO dataset and VOC PASCAL dataset
  * Support for Pelee304_VOC、Pelee304_COCO training and testing
  * Support for mulltigpu training
  * Support training and and testing in VOC and COCO  
  
### References
* [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882)
* [M2Det](https://github.com/qijiezhao/M2Det)
* [Pelee](https://github.com/Robert-JunWang/Pelee)