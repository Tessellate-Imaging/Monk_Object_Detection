export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh
. /usr/local/bin/virtualenvwrapper.sh


export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
workon detecto_rs && python Monk_Object_Detection/inference_engine/detecto_rs/lib/infer_folder_img_detecto_rs.py