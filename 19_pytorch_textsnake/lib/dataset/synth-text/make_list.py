import os
import scipy.io as io
from tqdm import tqdm

gt_mat_path = 'data/SynthText/gt.mat'
im_root = 'data/SynthText/'
txt_root = 'data/SynthText/gt/'

if not os.path.exists(txt_root):
    os.mkdir(txt_root)

print('reading data from {}'.format(gt_mat_path))
gt = io.loadmat(gt_mat_path)
print('Done.')

for i, imname in enumerate(tqdm(gt['imnames'][0])):
    imname = imname[0]
    img_id = os.path.basename(imname)
    im_path = os.path.join(im_root, imname)
    txt_path = os.path.join(txt_root, img_id.replace('jpg', 'txt'))

    if len(gt['wordBB'][0,i].shape) == 2:
        annots = gt['wordBB'][0,i].transpose(1, 0).reshape(-1, 8)
    else:
        annots = gt['wordBB'][0,i].transpose(2, 1, 0).reshape(-1, 8)
    with open(txt_path, 'w') as f:
        f.write(imname + '\n')
        for annot in annots:
            str_write = ','.join(annot.astype(str).tolist())
            f.write(str_write + '\n')

print('End.')