export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
workon gluoncv_finetune && python Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_center_net.py