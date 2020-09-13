## SynthText

SynthText is uploaded to Baidu Cloud [link](https://pan.baidu.com/s/17Gk301SwsnoESM1jQZRq0g), extract code `tb5g`

1. download from link above and unzip it SynthText.zip
2. transform `.mat` ground truth to `.txt` format in `gt`: `$ python dataset/synth-text/make_list.py`
3. make training list by running: `$ ls data/SynthText/gt/ > data/SynthText/image_list.txt` (see [issue #24](https://github.com/princewang1994/TextSnake.pytorch/issues/24))
4. pretrain using synthtext:

```bash
CUDA_VISIBLE_DEVICES=$GPUID python train.py synthtext_pretrain --dataset synth-text --viz --max_epoch 1 --batch_size 8
```