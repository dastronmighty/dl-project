import random
from torch.utils.data import DataLoader
from src.utils.datautils import WrappedDataLoader, CustomDataset, mount_to_device, seed_worker, get_jpgs_from_path

class Data:
    def __init__(self,
                 path,
                 val_percent=0.2,
                 test_amt=3000,
                 wrapped_function=None,
                 workers=0,
                 device='cpu',
                 batch_size=64,
                 verbose=False,
                 seed=42):
        self.dev = device
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose

        self.wrapped_function = mount_to_device
        if wrapped_function is not None:
            self.wrapped_function = wrapped_function

        paths = get_jpgs_from_path(path)
        random.seed(seed)
        random.shuffle(paths)

        self.N = len(paths)

        self.test_files = []
        for p in paths:
            if len(self.test_files) >= test_amt:
                break
            if "resized" in p:
                self.test_files.append(p)
                paths.remove(p)

        vl_amt = int(self.N * val_percent)
        tr_amt = self.N - vl_amt

        self.train_files = paths[0:tr_amt]
        self.val_files = paths[tr_amt:]


    def get_train_data(self):
        data = CustomDataset(self.train_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)

        return WrappedDataLoader(dl, lambda x, y: self.wrapped_function(x, y, self.dev))

    def get_val_data(self):
        data = CustomDataset(self.val_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, lambda x, y: self.wrapped_function(x, y, self.dev))

    def get_test_data(self):
        data = CustomDataset(self.test_files, self.dev)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, lambda x, y: self.wrapped_function(x, y, self.dev))
