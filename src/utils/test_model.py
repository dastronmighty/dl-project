import torch
from tqdm import tqdm
import numpy as np


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


def test_model(data,
               model,
               loss_func,
               logger):
    model.eval()
    test_data = data.get_test_data()
    losses, nums = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_data, "Test batch"):
            loss, n = loss_batch(model, xb, yb, loss_func, logger)
            losses.append(loss)
            nums.append(n)
    test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    logger.log_losses(test_loss, train=False)
    logger.print_epoch(-1)

