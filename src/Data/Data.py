import os

from torch.utils.data import DataLoader
from src.utils.datautils import WrappedDataLoader, CustomDataset, mountToDevice

class Data:
    def __init__(self, path,
                 val_amt=0.15,
                 test_amt=0.15,
                 workers=0,
                 device='cpu',
                 batch_size=64,
                 verbose=False,
                 reproduce=True):
        self.d_path = path
        self.dev = device
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose
        self.reproduce = reproduce

        files = os.listdir(path)
        N = len(files)
        v_amt, te_amt = int(N*val_amt), int(N*test_amt)
        tr_amt = N - v_amt - te_amt

        self.train_files = files[0:tr_amt]
        self.val_files = files[tr_amt:(tr_amt+v_amt)]
        self.test_files = files[(tr_amt+v_amt):]

    def get_train_data(self):
        data = CustomDataset(self.d_path, self.train_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=(not self.reproduce),
                        num_workers=self.workers)
        return WrappedDataLoader(dl, lambda x, y: mountToDevice(x, y, self.dev))

    def get_val_data(self):
        data = CustomDataset(self.d_path, self.val_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=(not self.reproduce),
                        num_workers=self.workers)
        return WrappedDataLoader(dl, lambda x, y: mountToDevice(x, y, self.dev))

    def get_test_data(self):
        data = CustomDataset(self.d_path, self.test_files , self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=(not self.reproduce),
                        num_workers=self.workers)
        return WrappedDataLoader(dl, lambda x, y: mountToDevice(x, y, self.dev))
