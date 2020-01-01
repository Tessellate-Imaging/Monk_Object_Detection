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

