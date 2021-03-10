import torch
import torchvision
import numpy as np
import random

from torch.utils.data import Dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(Dataset):
    def __init__(self, base, files, dev):
        self.device = dev
        self.base = base
        self.files = files

    def preprocess(self, file):
        y = int(file[-5:-4])
        img = torchvision.io.read_image(self.base+"/"+file)
        x = img.float()
        return x, y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.preprocess(self.files[index])


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def mountToDevice(X, Y, dev):
    return X.to(dev), Y.to(dev)

