wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MNr9dHBXHECChDgFW_FoPLw5NNBfME2P' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MNr9dHBXHECChDgFW_FoPLw5NNBfME2P" -O pretrained/resnet-101-0000.params && rm -rf /tmp/cookies.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1es3pEjamVlUM7SfvjthhnQqwryt7WXq4' -O pretrained/resnet-101-symbol.json


