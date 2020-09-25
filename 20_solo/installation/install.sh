cat requirements.txt | xargs -n 1 -L 1 pip install -vvvv

cd ../lib && pip install -vvvv .
