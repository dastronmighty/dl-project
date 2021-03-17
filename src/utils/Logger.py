import numpy as np
from src.utils.utils import init_folder


class Logger:

    def __init__(self,
                 log_name,
                 log_out,
                 metrics_dict,
                 train_early_stopping=False,
                 test_early_stopping=False,
                 stopping_attention=3,
                 overwrite=False,
                 verbose=False):
        """
        The Logger Class is used for recording training, validation and Testing
        :param log_name: the name of the current experiment
        :param log_out: the output directory to save the log to
        :param metrics_dict: a dictionary of the functions to use as metrics for the model
        :param train_early_stopping: whether to use early stopping on the training metric
        :param test_early_stopping: whether to use early stopping on the validation metric
        :param stopping_attention: the attention for early stopping (how far back to compare to)
        :param overwrite: whether to overwrite previous experiments with the same name
        :param verbose: wether or not to also print the logs
        """
        self.metrics_dict = metrics_dict
        self.verbose = verbose
        self.train_early_stopping = train_early_stopping
        self.test_early_stopping = test_early_stopping
        self.stopping_attention = stopping_attention

        self.train_history = {}
        self.test_history = {}
        for k in self.metrics_dict.keys():
            self.train_history[k] = []
            self.test_history[k] = []
        self.train_losses = []
        self.test_losses = []

        self.path = init_folder(log_name, log_out, overwrite)

        self.f = f"{self.path}/{log_name}.txt"

        self.curr_epoch = None
        self.curr_epoch_stats = None
        self.reset_curr_epoch_stats()
        with open(self.f, mode="w") as log_file:
            log_file.write("")

    def reset_curr_epoch_stats(self):
        """
        set all stats to empty
        """
        self.curr_epoch_stats = {
            "train": {},
            "test": {}
        }
        for k in self.metrics_dict.keys():
            self.curr_epoch_stats["train"][k] = []
            self.curr_epoch_stats["test"][k] = []

    def log_batch(self, epoch, xb, yb, train=True):
        """
        Log a batch of training
        :param epoch: the epoch
        :param xb: the predicted lables
        :param yb: the true lables
        :param train: whether or not this is from training
        """
        if self.curr_epoch != epoch: # if the epoch has changed
            self.reset_curr_epoch_stats()
            self.curr_epoch = epoch
        for k in self.metrics_dict.keys():
            met = self.metrics_dict[k](yb, xb)
            if train:
                self.curr_epoch_stats["train"][k].append(met)
            else:
                self.curr_epoch_stats["test"][k].append(met)

    def check_early_stopping(self):
        """
        Check if the model should stop training if the loss is going up
        """
        tr_es, te_es = False, False
        if len(self.train_losses) > (self.stopping_attention + 1): # only run once enough epochs have been run to check against
            if self.train_early_stopping:
                tr_es = self.check_early_stopping_helper()
            if self.test_early_stopping:
                te_es = self.check_early_stopping_helper(False)
        ret = (tr_es or te_es)
        if ret:
            log = "Early Stopping!"
            if self.verbose:
                print(log)
            with open(self.f, mode="a") as log_file:
                log_file.write(f"{log}\n")
        return ret

    def check_early_stopping_helper(self, train=True):
        """
        Check early stopping helper
        :param train: whether or not to check against the training or validation data
        """
        x = self.train_losses
        if train is False:
            x = self.test_losses
        last_idx = len(x) - 1
        last_loss = x[last_idx]
        l_grt_last = 0
        for i in range(last_idx - self.stopping_attention, last_idx): # for each of the last x (where x is the attention) losses
            if last_loss >= x[i]:
                l_grt_last += 1
        if l_grt_last == self.stopping_attention:
            # if each of the last x losses were smaller than the most recent then stop
            # because the loss is not going down
            return True
        return False

    def log_losses(self, loss, train=True):
        """
        log losses from training
        :param loss: the loss to log
        :param train: whether it was training (True or Flase)
        """
        if train:
            self.train_losses.append(loss)
        else:
            self.test_losses.append(loss)

    def compress_test(self):
        """
        compress an epoch of test stats (each batch) into single numbers
        """
        test_f_key = list(self.curr_epoch_stats["test"].keys())[0]
        if len(self.curr_epoch_stats["test"][test_f_key]) != 0: # check there is test data
            for stat in self.curr_epoch_stats["test"].keys():
                mean_stat = np.nanmean(np.array(self.curr_epoch_stats["test"][stat]))
                self.test_history[stat].append(mean_stat)

    def compress_epoch_stats(self, epoch=0):
        """
        compress an epoch of stats (each batch) into single numbers
        """
        if epoch == 0:
            for stat in self.curr_epoch_stats["train"].keys():
                mean_stat = np.nanmean(np.array(self.curr_epoch_stats["train"][stat]))
                self.train_history[stat].append(mean_stat)
        self.compress_test()
        self.reset_curr_epoch_stats()

    def print_epoch(self, epoch, override_string=None):
        """
        print the epoch to the file (and to the )
        :param epoch: current epoch
        :param override_string: a string we can place at the start of the current log
        """
        self.compress_epoch_stats(epoch)
        log = ""
        if epoch != -1:
            log = f"epoch : {epoch}"
        stat_log = ""
        has_test_data = len(self.test_history[list(self.test_history.keys())[0]]) != 0
        for k in self.metrics_dict.keys(): # for each metric
            if epoch != -1:
                stat = self.train_history[k][-1]
                stat_log += f" - Train {k} : {stat}"
            if has_test_data:
                stat = self.test_history[k][-1]
                stat_log += f" - Test {k} : {stat}"
        loss_log = ""
        if epoch != -1:
            tr_loss = self.train_losses[-1]
            avg_tr_loss = np.nanmean(np.array(self.train_losses))
            loss_log += f"Train Loss : {tr_loss} - Avg. Train Loss : {avg_tr_loss}"
        if len(self.test_losses) != 0:
            te_loss = self.test_losses[-1]
            avg_te_loss = np.nanmean(np.array(self.test_losses))
            loss_log += f" - Test Loss : {te_loss}"
            if epoch != -1:
                loss_log += f" - Avg. Test Loss : {avg_te_loss}"
        log = f"{log} - {loss_log}{stat_log}"
        if override_string is not None:
            log = f"{override_string} - {loss_log}{stat_log}"
        if self.verbose:
            print(log)
        with open(self.f, mode="a") as log_file:
            log_file.write(f"{log}\n")

