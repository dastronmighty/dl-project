from matplotlib import pyplot as plt
from datetime import datetime

import shutil
import os

import torch
import random
import numpy


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def curr_time():
    ct = datetime.now().strftime("%Y%m%d%H%M%S")
    return ct


def tensorToLabels(Y):
    """Turn a tensor to a list of string labels
    :param Y: the tensor of labels
    """
    labels = ["Disease","Normal"]
    to_lab = []
    for y in Y:
        to_lab.append(labels[y])
    return to_lab


def show_dataset(X, Y):
    """
    Show 9 images from a given batch
    :param X: Batch X
    :param Y: Batch Y
    """
    to_show, l = X[0:9], tensorToLabels(Y[0:9])
    rows, cols = 3, 3
    f, axs = plt.subplots(rows, cols, figsize=((5 * cols), (5 * rows)))
    for i, x in enumerate(to_show):
        im = (x.permute(1, 2, 0)).int()
        idx1, idx2 = i // cols, i % cols
        axs[idx1, idx2].imshow(im)
        axs[idx1, idx2].set_title(l[i])
        axs[idx1, idx2].axis('off')
    return f


def init_folder(name, path, overwrite):
    """
    A handy function for making sure there are no upsets when creating the folders for the logs and checkpoints
    :param name: the name of the folder
    :param path: the path to create the folder in
    :param overwrite: wether to delete the folder with the same name in the path if one is found
    :return: the path to the created folder
    """
    if name in os.listdir(path):
        if overwrite:
            shutil.rmtree(f"{path}/{name}")
            os.mkdir(f"{path}/{name}")
        else:
            ct = curr_time()
            warn = f"Folder {name} already existed saving to "
            name = name + ct
            warn += name
            print(warn)
            os.mkdir(f"{path}/{name}")
    else:
        os.mkdir(f"{path}/{name}")
    return f"{path}/{name}"


def make_folder_if_not_there(name, path):
    if name not in os.listdir(path):
        os.mkdir(f"{path}/{name}")
    return f"{path}/{name}"
