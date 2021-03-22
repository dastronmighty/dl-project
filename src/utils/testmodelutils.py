import torch
import numpy as np
from tqdm import tqdm



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