import numpy as np
from tqdm import tqdm

from src.Data.Data import Data
from src.utils.utils import save_ckp, get_compact_stats

import torch

from sklearn import metrics

class FitModel:

    def __init__(self, mod_config, model, opt, loss_func, verbose=False):
        self.history = []
        self.verbose = verbose
        self.loss_func = loss_func

        data = Data(mod_config["data_path"],
                    workers=mod_config["num_workers"],
                    dev=mod_config["dev"],
                    batch_size=mod_config["batch_size"])

        self.train_dl = data.get_train_data()
        self.test_dl = data.get_test_data()

        epochs = mod_config["epochs"]
        model.to(mod_config["dev"])

        epoch_checkpoint = mod_config["epoch_ckp"]

        for epoch in range(epochs):
            train_loss = self.train_model(model, opt)
            val_loss, val_stats = self.evaluate_model(model)
            if self.verbose:
                output_e = f"\nEpoch - {epoch}, "
                output_e += f"Training Loss - {train_loss}, "
                output_e += f"Validation Loss - {val_loss}, "
                output_e += f"Average Validation Accuracy - {val_stats['avg_acc']}, "
                output_e += f"Average Validation AUC - {val_stats['avg_auc']}, "
                print(output_e)
                print(val_stats)
            self.history.append({"epoch": epoch,
                                 "train_loss":train_loss,
                                 "val_loss": val_loss,
                                 "val_stats": val_stats})
            if (epoch > 1):
                if epoch_checkpoint is not None:
                    if (epoch % epoch_checkpoint == 0):
                        p = f"checkpoints/{mod_config['name']}"
                        save_ckp(model, opt, self.history[(epoch-epoch_checkpoint):], mod_config["name"], epoch, p)


    def train_model(self, model, opt):
        model.train()
        losses, nums = [], []
        for xb, yb in tqdm(self.train_dl, "batch", disable=(not self.verbose)):
            loss, n = self.loss_batch(model, xb, yb, opt)
            losses.append(loss)
            nums.append(n)
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        return train_loss

    def evaluate_model(self, model):
        model.eval()
        losses, nums, stats = [], [], []
        with torch.no_grad():
            for xb, yb in tqdm(self.test_dl, "validation batch", disable=(not self.verbose)):
                loss, n = self.loss_batch(model, xb, yb)
                bstats = self.get_metrics(model, xb, yb)
                stats.append(bstats)
                losses.append(loss)
                nums.append(n)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_stats = get_compact_stats(stats)
        return val_loss, val_stats

    def loss_batch(self, model, xb, yb, opt=None):
        loss = self.loss_func(model(xb).flatten(), yb.float())
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), len(xb)

    def get_metrics(self, model, xb, yb):
        yh = model(xb)
        yhat = yh.detach().numpy().flatten()
        ytrue = yb.detach().numpy()
        acc = metrics.accuracy_score(ytrue, (yhat>0.5)*1)
        fpr, tpr, aucthresholds = metrics.roc_curve(ytrue, yhat, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        stats = {
            "auc": auc,
            "acc": acc,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": aucthresholds,
        }
        return stats
