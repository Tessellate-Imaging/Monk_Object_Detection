## Project Details
Pipeline based on GluonCV Fintuing project - https://gluon-cv.mxnet.io/build/examples_detection/index.html

<br />
<br />
<br />

## Installation

Supports 
- Python 3.6
- Python 3.7
    
`cd installation`

Check the cuda version using the command

`nvcc -V`

Select the right requirements file and run 

`cat <selected requirements file> | xargs -n 1 -L 1 pip install`

For example for cuda 9.0

`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`



## TODO
- [x] Add SSD support
- [x] Add YoloV3 support
- [x] Add Faster-RCNN support
- [ ] Test on Kaggle and Colab 
- [ ] Add validation feature & data pipeline
- [ ] Add Optimizer selection feature
- [ ] Enable Learning-Rate Scheduler Support
- [ ] Enable Layer Freezing
- [ ] Set Verbosity Levels
- [ ] Add Project management and version control support (Similar to Monk Classification)
- [ ] Add Graph Visualization Support
- [ ] Enable batch proessing at inference
- [ ] Add feature for top-k output visualization
- [ ] Add Multi-GPU training
