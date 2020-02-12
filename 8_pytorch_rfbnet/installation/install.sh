cat requirements.txt | xargs -n 1 -L 1 pip install

cd ../lib/ && chmod +x make.sh && ./make.sh