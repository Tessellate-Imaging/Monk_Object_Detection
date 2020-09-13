import scipy.io as io
import numpy as np
import os

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class DeployDataset(TextDataset):

    def __init__(self, image_root, transform=None):
        super().__init__(transform)
        self.image_root = image_root
        self.image_list = os.listdir(image_root)

    def __getitem__(self, item):

        # Read image data
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        image = pil_load_img(image_path)

        return self.get_test_data(image, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation
    from torch.utils.data import DataLoader

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = BaseTransform(
        size=512, mean=means, std=stds
    )

    trainset = DeployDataset(
        image_root='data/total-text/Images/Train',
        transform=transform
    )

    loader = DataLoader(trainset, batch_size=1, num_workers=0)

    for img, meta in loader:
        print(img.size(), type(img))