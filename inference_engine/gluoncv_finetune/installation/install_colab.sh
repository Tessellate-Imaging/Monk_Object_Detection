#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 gluoncv_finetune

workon gluoncv_finetune && cat Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/requirements_colab.txt | xargs -n 1 -L 1 pip install 

echo "Completed"