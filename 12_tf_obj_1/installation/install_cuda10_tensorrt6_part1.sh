cat requirements_cuda10_tensorrt6.txt | xargs -n 1 -L 1 pip install

pip3 install keras_applications==1.0.8 --no-deps

pip3 install keras_preprocessing==1.0.8 --no-deps