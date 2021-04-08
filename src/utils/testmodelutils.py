import os

import torch
import numpy as np
from tqdm import tqdm

from src.Data.Data import Data

from src.utils.Plotting import show_test_on_images
from src.utils.datautils import sample_from_data_loader
from src.utils.utils import get_final_ckps


def loss_batch(model, xb, yb, loss_func, logger):
    xb = model(xb).flatten()
    yb = yb.float()
    loss = loss_func(xb, yb.float())
    with torch.no_grad():
        xb = xb.cpu()
        yb = yb.cpu()
        xb = xb.detach().numpy()
        yb = yb.detach().numpy()
        logger.log_batch(-1, xb, yb, train=False)
    return loss.item(), len(xb)


def test_model(test_data,
               model,
               loss_func,
               logger,
               name):
    """
    Test a model on test data
    :param test_data: the test data
    :param model: the model to test
    :param loss_func: the loss function to use
    :param logger: the logger for recording the test
    :param name: the name of the experiment
    """
    model.eval()
    losses, nums = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_data, "Test batch"):
            loss, n = loss_batch(model, xb, yb, loss_func, logger)
            losses.append(loss)
            nums.append(n)
    test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    logger.log_losses(test_loss, train=False)
    logger.print_epoch(-1, override_string=f"Final Stats from {name}")


def get_test_64batch_from_path(path, wrapped=None, dev="cpu", seed=42):
    data = Data(path,
                augmented=False,
                workers=0,
                device=dev,
                test_amt=1000,
                batch_size=64,
                wrapped_function=wrapped,
                seed=seed)
    x, y = sample_from_data_loader(data.get_test_data())
    return x, y


def get_final_ckps(dir_name):
    ckps = []
    for _ in os.listdir(dir_name):
        p = f"{dir_name}/{_}"
        if os.path.isdir(p):
            u_ps = get_final_ckps(p)
            ckps += u_ps
        elif ".pt" in p and ("final" in p.lower()):
            ckps.append(p)
    return ckps


def test_models_on_batch_and_show(expt_name,
                                  data_path,
                                  ckps_path,
                                  model_class,
                                  model_kwargs,
                                  dev=torch.device("cpu"),
                                  wrapped=None,
                                  rows=3,
                                  cols=3,
                                  seed=42):
    fin_ckps = get_final_ckps(ckps_path)
    for ckp in fin_ckps:
        show_test_on_images(expt_name,
                            data_path,
                            ckp, model_class,
                            model_kwargs,
                            dev=dev,
                            wrapped=wrapped,
                            rows=rows,
                            cols=cols,
                            seed=seed)