import numpy as np
from tqdm import tqdm
import torch
import random

from src.utils.utils import set_seed

class FitModel:

    def __init__(self,
                 model,
                 data,
                 opt,
                 loss_func,
                 epochs,
                 dev,
                 logger,
                 checkpointer,
                 verbose=False,
                 seed=42):
        """
        The main class used to actually train models
        :param model: the model to train
        :param data: data to use (Of type Data from this package)
        :param opt: optimizer to use
        :param loss_func: loss function to use
        :param epochs: number of epochs to train for
        :param dev: what device to use
        :param logger: a logger to record training
        :param checkpointer: a checkpointer to save the trained (and mid training) model(s)
        :param verbose: the verbosity of training
        :param seed: the seed to use for reproducibility
        """
        set_seed(self.seed)
        self.train_dl = data.get_train_data()
        self.val_dl = data.get_val_data()
        self.loss_func = loss_func

        model = model.to(dev)

        self.logger = logger
        self.verbose = verbose

        for epoch in range(epochs):
            self.train_model(epoch, model, opt)
            self.evaluate_model(epoch, model)
            self.logger.print_epoch(epoch)
            checkpointer.save(epoch, model, opt)
            if logger.check_early_stopping():
                break
        checkpointer.save_override(-1, model, add_tag="FINAL")

    def train_model(self, epoch, model, opt):
        model.train()
        losses, nums = [], []
        for xb, yb in tqdm(self.train_dl, "training batch", disable=(not self.verbose)):
            loss, n = self.loss_batch(epoch, model, xb, yb, opt)
            losses.append(loss)
            nums.append(n)
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        self.logger.log_losses(train_loss)

    def evaluate_model(self, epoch, model):
        model.eval()
        losses, nums = [], []
        with torch.no_grad():
            for xb, yb in tqdm(self.val_dl, "validation batch", disable=(not self.verbose)):
                loss, n = self.loss_batch(epoch, model, xb, yb, train=False)
                losses.append(loss)
                nums.append(n)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        self.logger.log_losses(val_loss, train=False)

    def loss_batch(self, epoch, model, xb, yb, opt=None, train=True):
        xb = model(xb).flatten()
        yb = yb.float()
        loss = self.loss_func(xb, yb.float())
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            xb = xb.cpu()
            yb = yb.cpu()
            xb = xb.detach().numpy()
            yb = yb.detach().numpy()
            self.logger.log_batch(epoch, xb, yb, train=train)
        return loss.item(), len(xb)
