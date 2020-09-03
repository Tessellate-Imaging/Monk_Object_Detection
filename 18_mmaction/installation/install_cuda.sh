cat requirements.txt | xargs -n 1 -L 1 pip install -vvvv

cd ../lib/mmcv && MMCV_WITH_OPS=1 pip install -vvvv -e .

cd ../lib/ && pip install -vvvv -e .
