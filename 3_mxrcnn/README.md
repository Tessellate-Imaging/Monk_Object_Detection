Project Details

Pipeline based on MX-RCNN project - https://github.com/ijkguo/mx-rcnn



Installation

Supports

    Python 3.6
    Cuda-9.0 (Other cuda version support are experimental)

cd installation

Check the cuda version using the command

nvcc -V

Select the right requirements file and run

cat <selected requirements file> | xargs -n 1 -L 1 pip install

For example for cuda 9.0

cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install



Pipeline

    Load Dataset

`set_dataset_params(root_dir="../sample_dataset/", 
                   coco_dir="kangaroo", imageset="Images");`

    Load Model

gtf.Model(model_name, use_pretrained=pretrained, use_gpu=gpu);

    Set Hyper-parameter

gtf.Set_Learning_Rate(0.001);

    Train

gtf.Train(epochs, params_file);
