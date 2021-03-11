import random
from torch.utils.data import DataLoader
from src.utils.datautils import WrappedDataLoader, CustomDataset, mount_to_device, seed_worker, get_jpgs_from_path

class Data:
    def __init__(self,
                 path,
                 total_amt=3000,
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
            self.wrapped_function = lambda x, y: mount_to_device(*wrapped_function(x, y), self.dev)

        paths = get_jpgs_from_path(path)

        regulars = [i for i in paths if "resized" in i]
        for p in regulars:
            paths.remove(p)

        self.test_files = regulars[0:test_amt]
        self.train_files = []
        self.val_files = []

        regulars = regulars[test_amt:]
        reg_left = int(len(regulars)/2)

        self.train_files += regulars[0:reg_left]
        self.val_files += regulars[reg_left:]

        random.seed(seed)
        random.shuffle(paths)
        total_amt = total_amt - len(regulars)
        paths = paths[0:total_amt]
        self.N = len(paths)

        tr_amt = self.N - int(self.N * val_percent)
        self.train_files += paths[0:tr_amt]
        self.val_files += paths[tr_amt:]

        if verbose:
            for type in ["autocontrast", "equalize", "invert", "resized", "rotated"]:
                ltr = len([i for i in self.train_files if type in i])
                print(f"# of {type} in Train = {ltr}")
                lv = len([i for i in self.val_files if type in i])
                print(f"# of {type} in Validation = {lv}")
                lte = len([i for i in self.test_files if type in i])
                print(f"# of {type} in Test = {lte}")
            print(f"Total size of Train = {len(self.train_files)}")
            print(f"Total size of Validation = {len(self.val_files)}")
            print(f"Total size of Test = {len(self.test_files)}")



    def get_train_data(self):
        data = CustomDataset(self.train_files)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, self.wrapped_function)

    def get_val_data(self):
        data = CustomDataset(self.val_files)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, self.wrapped_function)

    def get_test_data(self):
        data = CustomDataset(self.test_files)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, self.wrapped_function)
