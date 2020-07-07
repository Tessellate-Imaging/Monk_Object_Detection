#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 efficientdet_pytorch

workon efficientdet_pytorch && cat Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/requirements_cpu.txt | xargs -n 1 -L 1 pip install 

echo "Completed"