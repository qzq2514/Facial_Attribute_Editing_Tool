# ------------------------------------------------------------
# Real-Time Style Transfer Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ------------------------------------------------------------
import os
import cv2
import sys
import scipy.misc
import numpy as np
from joblib import Parallel, delayed


def get_edge_core(image):
    #image=image+10
    image[image>255]=255
    kernel_size = 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.Canny(np.uint8(image), 50, 150)
    dilation = cv2.dilate(edges, kernel)  # 对边缘(白色)膨胀
    dilation = np.array([dilation,dilation,dilation])
    return np.transpose(dilation, (1, 2, 0))

def get_edge(batch_images):
    num_job = np.shape(batch_images)[0]  # batch_size
    batch_out = Parallel(n_jobs=num_job)(delayed(get_edge_core)(image) for image in batch_images)
    return np.array(batch_out)


def imread(path, is_gray_scale=True, img_size=None):
    if is_gray_scale:
        img = scipy.misc.imread(path, flatten=True).astype(np.float32)
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float32)

    if not (img.ndim == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def exists(p, msg):
    assert os.path.exists(p), msg


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()

