import random
from torch.utils.data import DataLoader
from src.utils.datautils import WrappedDataLoader, CustomDataset, mount_to_device, seed_worker, get_jpgs_from_path
from tqdm import tqdm
import torch
import numpy as np

from src.utils.utils import set_seed


class Data:

    def __init__(self,
                 path,
                 augmented,
                 total_amt=16384,
                 val_percent=0.25,
                 test_amt=768,
                 wrapped_function=None,
                 workers=0,
                 device=torch.device('cpu'),
                 batch_size=64,
                 verbose=False,
                 seed=42):
        """
        An object to Take Data from a path and convert it to tensors
        :param path: Path to Data
        :param augmented: whether the data is augmented or not
        :param total_amt: the total amount of data to use for Training and Validation
        :param val_percent: the percentage of data to use for Validation
        :param test_amt: the amount of Data to set aside for Testing
        :param wrapped_function: a function to apply to the data if required
        :param workers: the number of workers to use
        :param device: the device to use
        :param batch_size: the batch size to use
        :param verbose: the verbosity to use
        :param seed: the seed to use for reproducibility
        """
        self.device = device
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose
        self.seed = seed
        set_seed(self.seed)

        self.wrapped_function = lambda x, y: mount_to_device(x, y, self.device)
        if wrapped_function is not None:
            self.wrapped_function = lambda x, y: mount_to_device(*wrapped_function(x, y), self.device)

        self.train_files, self.val_files, self.test_files = [], [], []
        if augmented:
            self.get_augmented_sets(path, total_amt, val_percent, test_amt)
        else:
            self.get_non_augmented_sets(path, total_amt, val_percent, test_amt)

        total = self.train_files + self.val_files + self.test_files
        if verbose:
            pt, nt = self.calc_distribution(self.train_files)
            pv, nv = self.calc_distribution(self.val_files)
            pte, nte = self.calc_distribution(self.test_files)
            print(f"Total Size = {len(total)}")
            print(f"Total size of Train = {len(self.train_files)} (pos = {pt}, neg = {nt})")
            print(f"Total size of Validation = {len(self.val_files)} (pos = {pv}, neg = {nv})")
            print(f"Total size of Test = {len(self.test_files)} (pos = {pte}, neg = {nte})")
            if augmented:
                for type in ["autocontrast", "equalize", "invert", "resized", "rotated"]:
                    ltr = len([i for i in self.train_files if type in i])
                    print(f"# of {type} in Train = {ltr}")
                    lv = len([i for i in self.val_files if type in i])
                    print(f"# of {type} in Validation = {lv}")
                    lte = len([i for i in self.test_files if type in i])
                    print(f"# of {type} in Test = {lte}")
            print("Checking for duplicates...")
        if len(total) != len(set(total)):
            raise RuntimeError("Something has gone wrong! there are duplicates in data")
        else:
            if verbose:
                print("There are no duplicates in data!")

    def get_non_augmented_sets(self, dir, total_amt, val_percent, test_amt):
        """
        Function to get the datasets for non-augmented data
        :param dir: directory to data
        :param total_amt: the total amount of data to use for Training and Validation
        :param val_percent: the percentage of data to use for Validation
        :param test_amt: the amount of Data to set aside for Testing
        """
        paths = self.get_paths_from_dir(dir)
        len_p = len(paths)
        test_amt = test_amt if (test_amt < (len_p * 0.5)) else int(len_p * 0.2)
        total_amt = total_amt if (len_p >= total_amt + test_amt) else (len_p - test_amt)
        self.test_files = paths[0:test_amt]
        paths = paths[test_amt:]
        trn_amt = total_amt - int(total_amt * val_percent)
        self.train_files, self.val_files = paths[0:trn_amt], paths[trn_amt:]

    def get_augmented_sets(self, dir, total_amt, val_percent, test_amt):
        """
        Function to get the datasets for augmented data
        :param dir: directory to data
        :param total_amt: the total amount of data to use for Training and Validation
        :param val_percent: the percentage of data to use for Validation
        :param test_amt: the amount of Data to set aside for Testing
        """
        pos_p, neg_p, unaugmented = self.get_augmented_paths(dir)
        train_files, val_files, self.test_files = self.put_unaugmented(unaugmented, test_amt, val_percent)
        self.train_files, self.val_files = self.split_file_paths(pos_p, neg_p, train_files, val_files, total_amt,
                                                                 val_percent)

    def get_paths_from_dir(self, dir):
        """
        :param dir: directory to data
        :return: the absolute paths for the data
        """
        paths = get_jpgs_from_path(dir)
        random.shuffle(paths)
        return paths

    def split_into_p_and_n(self, paths):
        """
        Split Data paths into positive and negative paths
        :param paths: A list of paths to split
        :return: two lists one of positve imgs and one of negative imgs
        """
        pos_samples, neg_samples = [], []
        for p in tqdm(paths, "Splitting", disable=(not self.verbose)):
            if p[-5:-4] == "1":
                pos_samples.append(p)
            if p[-5:-4] == "0":
                neg_samples.append(p)
        return pos_samples, neg_samples

    def calc_distribution(self, paths):
        """
        calculate the number of each class in dataset
        :param paths: the array of paths to calculate on
        :return: the number of positive and negatives in the paths list
        """
        pos, neg = self.split_into_p_and_n(paths)
        return len(pos), len(neg)

    def get_augmented_paths(self, dir):
        """
        Get the paths for the augmented data
        :param dir: The directory to get the data from
        :return: positive paths, negative paths, and unaugmented data
        """
        paths = self.get_paths_from_dir(dir)
        if self.verbose:
            print(f"Pulling out un-augmented imgs")
        unaugmented = [i for i in tqdm(paths, "Extracting", disable=(not self.verbose)) if "resized" in i]
        for p in tqdm(unaugmented, "Removing", disable=(not self.verbose)):
            paths.remove(p)
        if self.verbose:
            print(f"Stabiliszing Dataset")
        pos_samples, neg_samples = self.split_into_p_and_n(paths)
        return pos_samples, neg_samples, unaugmented

    def split_file_paths(self, pos_p, neg_p, train_files, val_files, t_amt, v_p):
        """
        Split the file paths into the train files and validation files and make sure that
        the number of positive and negatives are balanced.
        :param pos_p: the positive paths
        :param neg_p: the negative paths
        :param train_files: the current train files
        :param val_files: the current validation files
        :param t_amt: the total amount of data to use
        :param v_p: the percent of data to use for validation
        :return: train and validation files
        """
        v_amt = int(t_amt * v_p)
        tr_amt = t_amt - v_amt
        trp, trn = self.calc_distribution(train_files)
        vp, vn = self.calc_distribution(val_files)
        pos_for_train = (int(tr_amt/2)-trp) if (trp < int(tr_amt/2)) else 0
        neg_for_train = (int(tr_amt/2)-trn) if (trn < int(tr_amt/2)) else 0
        train_files += pos_p[0:pos_for_train]
        train_files += neg_p[0:neg_for_train]
        pos_for_val = (int(v_amt/2)-vp) if (vp < int(v_amt/2)) else 0
        neg_for_val = (int(v_amt/2)-vn) if (vn < int(v_amt/2)) else 0
        val_files += pos_p[pos_for_train:pos_for_train+pos_for_val]
        val_files += neg_p[neg_for_train:neg_for_train+neg_for_val]
        return train_files, val_files

    def put_unaugmented(self, unaugmented, test_amt, vp):
        """
        Put unaugmented files into the test train, test, and validation lists
        :param unaugmented: the unagmented data array
        :param test_amt: the amount of data to use for test data
        :param vp: the validation percent
        :return: train, validation, and test files
        """
        test_files = unaugmented[0:test_amt]
        unaugmented = unaugmented[test_amt:]
        lau = len(unaugmented)
        tr_ua = lau - int(lau * vp)
        train_files = unaugmented[0:tr_ua]
        val_files = unaugmented[tr_ua:]
        return train_files, val_files, test_files

    def get_train_data(self):
        """
        Get the Training Data Loader
        :return: The Wrapped Data Loader
        """
        set_seed(self.seed)
        data = CustomDataset(self.train_files)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, self.wrapped_function)

    def get_val_data(self):
        """
        Get the Validation Data Loader
        :return: The Wrapped Data Loader
        """
        set_seed(self.seed)
        data = CustomDataset(self.val_files)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, self.wrapped_function)

    def get_test_data(self):
        """
        Get the Testing Data Loader
        :return: The Wrapped Data Loader
        """
        set_seed(self.seed)
        data = CustomDataset(self.test_files)
        dl = DataLoader(data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.workers,
                        worker_init_fn=seed_worker)
        return WrappedDataLoader(dl, self.wrapped_function)
