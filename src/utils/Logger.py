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
        self.curr_epoch_stats = {
            "train": {},
            "test": {}
        }
        for k in self.metrics_dict.keys():
            self.curr_epoch_stats["train"][k] = []
            self.curr_epoch_stats["test"][k] = []

    def log_batch(self, epoch, xb, yb, train=True):
        if self.curr_epoch != epoch:
            self.reset_curr_epoch_stats()
            self.curr_epoch = epoch
        for k in self.metrics_dict.keys():
            met = self.metrics_dict[k](yb, xb)
            if train:
                self.curr_epoch_stats["train"][k].append(met)
            else:
                self.curr_epoch_stats["test"][k].append(met)

    def check_early_stopping(self):
        tr_es, te_es = False, False
        if len(self.train_losses) > (self.stopping_attention + 1):
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
        x = self.train_losses
        if train is False:
            x = self.test_losses
        last_idx = len(x) - 1
        last_loss = x[last_idx]
        l_grt_last = 0
        for i in range(last_idx - self.stopping_attention, last_idx):
            if last_loss > x[i]:
                l_grt_last += 1
        if l_grt_last == self.stopping_attention:
            return True
        return False

    def log_losses(self, loss, train=True):
        if train:
            self.train_losses.append(loss)
        else:
            self.test_losses.append(loss)

    def compress_test(self):
        test_f_key = list(self.curr_epoch_stats["test"].keys())[0]
        if len(self.curr_epoch_stats["test"][test_f_key]) != 0:
            for stat in self.curr_epoch_stats["test"].keys():
                mean_stat = np.nanmean(np.array(self.curr_epoch_stats["test"][stat]))
                self.test_history[stat].append(mean_stat)

    def compress_epoch_stats(self, epoch=0):
        if epoch == 0:
            for stat in self.curr_epoch_stats["train"].keys():
                mean_stat = np.nanmean(np.array(self.curr_epoch_stats["train"][stat]))
                self.train_history[stat].append(mean_stat)
        self.compress_test()
        self.reset_curr_epoch_stats()

    def print_epoch(self, epoch, override=None):
        self.compress_epoch_stats(epoch)
        log = ""
        if epoch != -1:
            log = f"epoch : {epoch}"
        stat_log = ""
        has_test_data = len(self.test_history[list(self.test_history.keys())[0]]) != 0
        for k in self.metrics_dict.keys():
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
        if override is not None:
            log = f"{override} - {loss_log}{stat_log}"
        if self.verbose:
            print(log)
        with open(self.f, mode="a") as log_file:
            log_file.write(f"{log}\n")

