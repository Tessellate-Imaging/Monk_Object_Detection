cat requirements_colab.txt | xargs -n 1 -L 1 pip install

cd ../lib/core/models/py_utils/_cpools/ && python setup.py install --user

cd ../../../external && make