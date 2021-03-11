import random
from torch.utils.data import DataLoader
from src.utils.datautils import WrappedDataLoader, CustomDataset, mountToDevice, seed_worker, get_jpgs_from_path

class Data:
    def __init__(self, path,
                 val_amt=0.15,
                 test_amt=0.15,
                 workers=0,
                 device='cpu',
                 batch_size=64,
                 verbose=False,
                 seed=42):
        self.dev = device
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose

        paths = get_jpgs_from_path(path)
        random.seed(seed)
        random.shuffle(paths)

        N = len(paths)
        v_amt, te_amt = int(N * val_amt), int(N * test_amt)
        tr_amt = N - v_amt - te_amt
        self.train_files = paths[0:tr_amt]
        self.val_files = paths[tr_amt:(tr_amt + v_amt)]
        self.test_files = paths[(tr_amt + v_amt):]

    def get_train_data(self):
        data = CustomDataset(self.d_path, self.train_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)

        return WrappedDataLoader(dl, lambda x, y: mountToDevice(x, y, self.dev))

    def get_val_data(self):
        data = CustomDataset(self.d_path, self.val_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, lambda x, y: mountToDevice(x, y, self.dev))

    def get_test_data(self):
        data = CustomDataset(self.d_path, self.test_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, lambda x, y: mountToDevice(x, y, self.dev))
