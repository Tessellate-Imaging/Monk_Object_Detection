#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh
. /usr/local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 detecto_rs

workon detecto_rs && cat Monk_Object_Detection/inference_engine/detecto_rs/installation/requirements_cuda10.1.txt | xargs -n 1 -L 1 pip install 

workon detecto_rs && cd Monk_Object_Detection/inference_engine/detecto_rs/lib/ && pip install -v -e .

echo "Completed"