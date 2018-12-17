import numpy as np
from PIL import Image
import random
import scipy.misc


def load(files, crop=148, size=64):
    imgs = []
    for f in files:
        i = np.array(Image.open(f))

        h, w, _ = i.shape
        top = (h - crop)//2
        left = (w - crop)//2
        i = scipy.misc.imresize(i[top:top+crop, left:left+crop, :],
                                [size, size]).astype(np.float32)

        i *= 2/255.
        i -= 1.
        imgs.append(i.transpose((2, 0, 1)))
    return np.array(imgs)
