from PIL import Image
import numpy as np


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image