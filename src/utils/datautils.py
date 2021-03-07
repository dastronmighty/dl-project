import torch
import torchvision
from numpy import asarray
from PIL import Image

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, base, files, dev):
        self.device = dev
        self.base = base
        self.files = files

    def preprocess(self, file):
        Y = int(file[-5:-4])
        img = torchvision.io.read_image(self.base+"/"+file)
        img = torchvision.transforms.functional.rgb_to_grayscale(img)
        img = img/255
        X = img.float()
        return X, Y

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
            yield (self.func(*b))

def mountToDevice(X, Y, dev):
    return X.to(dev), Y.to(dev)