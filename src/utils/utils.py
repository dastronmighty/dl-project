from matplotlib import pyplot as plt
from datetime import datetime
from torchvision import transforms
import shutil
import os


def current_datetime():
    ct = datetime.now().strftime("%Y%m%d%H%M%S")
    return ct


def tensorToLabels(Y):
    labels = ["Disease", "Normal"]
    to_lab = []
    for y in Y:
        to_lab.append(labels[y])
    return to_lab


def show_dataset(X, Y):
    to_show, l = X[0:9], tensorToLabels(Y[0:9])
    rows, cols = 3, 3
    f, axs = plt.subplots(rows, cols, figsize=((5 * cols), (5 * rows)))
    for i, x in enumerate(to_show):
        im = ((x.permute(1, 2, 0))*255).int()
        idx1, idx2 = i // cols, i % cols
        axs[idx1, idx2].imshow(im)
        axs[idx1, idx2].set_title(i)
        axs[idx1, idx2].axis('off')
    return f


def init_folder(name, path, overwrite):
    if name in os.listdir(path):
        if overwrite:
            shutil.rmtree(f"{path}/{name}")
            os.mkdir(f"{path}/{name}")
        else:
            ct = current_datetime()
            warn = f"Folder {name} already existed saving to "
            name = name + ct
            warn += name
            print(warn)
            os.mkdir(f"{path}/{name}")
    else:
        os.mkdir(f"{path}/{name}")
    return f"{path}/{name}"


def resize_wrapper(x, y, s):
    x = transforms.functional.resize(x, size=(s, s))
    return x, y


def get_resize_wrapper(size):
    return lambda x, y: resize_wrapper(x, y, size)

