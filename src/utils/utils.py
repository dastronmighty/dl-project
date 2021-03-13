from src.Data.Data import Data
from src.utils.Metrics import auc, acc

from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm

import shutil
import os

import numpy as np

import torch
from torchvision import transforms

def curr_time():
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
            ct = curr_time()
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


def test_model_on_one_batch(epochs, model, m_kwargs, p, wrapped):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print("Loading Data...")
    data = Data(p, wrapped_function=wrapped, device=device)
    x, y = next(iter(data.get_train_data()))
    mod = model(**m_kwargs)
    mod = mod.to(device)
    opt = torch.optim.Adam(params=mod.parameters(), lr=0.00025)
    loss_func = torch.nn.BCELoss()
    mets = {
        "loss": [],
        "acc": [],
        "auc": []
    }
    for i in tqdm(range(epochs), "epoch"):
        y_hat = mod(x)
        y_hat = y_hat.flatten()
        loss = loss_func(y_hat, y.float())
        mets["loss"].append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
        mets["auc"].append(acc(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()))
        mets["acc"].append(auc(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()))
        print(f"Loss = {mets['loss'][-1]} - acc = {mets['acc'][-1]} - auc = {mets['auc'][-1]}")
    print("")
    print(f"Min loss reached {np.min(mets['loss'])} - reach on epoch {np.argmin(mets['loss'])}")
    print(f"Max accuracy reached {np.max(mets['acc'])} - reach on epoch {np.argmax(mets['acc'])}")
    print(f"Max AUC reached {np.max(mets['auc'])} - reach on epoch {np.argmax(mets['auc'])}")


