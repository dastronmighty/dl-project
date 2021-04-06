from src.Data.Data import Data
from src.utils.Metrics import auc, acc
from src.utils.testmodelutils import test_model
from src.utils.Checkpoint import load_ckp
from src.utils.Logger import Logger

import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn

from tqdm import tqdm

from collections import OrderedDict

import os


def resize(x, y, s):
    """
    Resize X to s x s
    :param x: 2d tensor to resize
    :param y: labels
    :param s: size to resize to
    :return: x, y
    """
    x = transforms.functional.resize(x, size=(s, s))
    return x, y


def get_resize_wrapper(size):
    """
    a quick helper for making a wrapper function for resizing the data
    :param size: the size to resize to
    :return: a function that will resize X and give back x, y (as required)
    """
    return lambda x, y: resize(x, y, size)


def summarise_model(model, size, bs):
    # Less stressful interface to model summary
    mod = model()
    summ, _ = summary(mod, size, bs)
    return summ


def test_model_on_one_batch(epochs, model, p, wrapped, m_kwargs, data_kwargs):
    """
    Chcke how a model does on one batch of data (mostly used to check for bugs so as to not run entire training sessions)
    it is also used to see if a model can fit to the one batch of data
    :param epochs: number of Epochs to train for
    :param model: the model class to use
    :param m_kwargs: the mosel arguments to use
    :param p: the path to the data
    :param wrapped: a function to use on the data if required (pass None if not)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print("Loading Data...")
    data = Data(p, wrapped_function=wrapped, device=device, **data_kwargs)
    x, y = next(iter(data.get_train_data()))
    mod = model(**m_kwargs)
    mod = mod.to(device)
    opt = torch.optim.Adam(params=mod.parameters(), lr=0.00025)
    loss_func = torch.nn.BCELoss()
    mets = {
        "loss": [],
        "acc": [],
        "auc": []
    }
    for i in tqdm(range(epochs), "epoch"):
        y_hat = mod(x)
        y_hat = y_hat.flatten()
        loss = loss_func(y_hat, y.float())
        mets["loss"].append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
        mets["auc"].append(acc(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()))
        mets["acc"].append(auc(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()))
        print(f"Loss = {mets['loss'][-1]} - acc = {mets['acc'][-1]} - auc = {mets['auc'][-1]}")
    print("")
    print(f"Min loss reached {np.min(mets['loss'])} - reach on epoch {np.argmin(mets['loss'])}")
    print(f"Max accuracy reached {np.max(mets['acc'])} - reach on epoch {np.argmax(mets['acc'])}")
    print(f"Max AUC reached {np.max(mets['auc'])} - reach on epoch {np.argmax(mets['auc'])}")


def summary(model, input_size, batch_size=2, device=torch.device('cpu'), dtypes=None):
    # summaries model
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    return result, params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    """
    a pytorch version of TF's Summary() funciton
    This is taken from this package:

    The package for some reason only prints out the summary but I wanted to get the string so I yoinked out the code
    and made it return the string
    :param model: a model to show the summary of
    :param input_size: the required input size (as a shape tuple) of the model (batch size excluded)
    :param batch_size: the batch size to show
    :param device: device to use
    :param dtypes: if other dtypes are required
    :return: the summary string of the model
    """
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)


def test_ckps(data_dir,
              auged,
              ckp_dir,
              log_dir,
              model,
              model_kwargs,
              mets,
              device=torch.device("cpu"),
              loss_func=torch.nn.BCELoss(),
              total_amt=16384,
              val_percent=0.25,
              test_amt=768,
              wrapped_function=None,
              workers=0,
              seed=42):
    """
    :param data_dir: the directory to the data
    :param auged: whether or not the data is augmented or not
    :param ckp_dir: the directory to the checkpoint
    :param log_dir: the directory to the logs
    :param model: the class of the model
    :param mets: the metrics to use
    :param device: the device to use
    :param loss_func: the loss function to use
    :param total_amt: the total amount of data to use
    :param val_percent: the percent of data to use for validation
    :param test_amt: the amount of data to use for testing
    :param wrapped_function: the wrapped function ot use
    :param workers: the number of workers to use
    :param seed: the seed to use
    """
    data = Data(data_dir,
                auged,
                total_amt=total_amt,
                val_percent=val_percent,
                test_amt=test_amt,
                wrapped_function=wrapped_function,
                workers=workers,
                device=device,
                verbose=True,
                seed=seed)
    test_data = data.get_test_data()
    for dir_ in os.listdir(ckp_dir):
        mod_ckp_dir = f"{ckp_dir}/{dir_}"
        if os.path.isdir(mod_ckp_dir):
            for ckp in os.listdir(mod_ckp_dir):
                if "FINAL" in ckp:
                    fin_dir = f"{mod_ckp_dir}/{ckp}"
                    name = ckp.replace(".pt", "")
                    logger = Logger(f"{name}_TEST",
                                    log_dir,
                                    mets,
                                    overwrite=True,
                                    verbose=True)
                    mod = model(**model_kwargs)
                    mod.to(device)
                    print(f"Loading model from {fin_dir}")
                    mod, _ = load_ckp(fin_dir, mod, dev=device)
                    name = name.replace("_FINAL.pt", "")
                    test_model(test_data, mod, loss_func, logger, name)

def test_model_checkpoint(ckp_path, model_class, model_kwargs, resize_size, data_path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = get_resize_wrapper(resize_size)
    print("Loading Data...")
    data = Data(data_path,
                augmented=True,
                device=dev,
                batch_size=32,
                wrapped_function=wrapper,
                verbose=True)
    print("Loading Model...")
    mod = model_class(**model_kwargs)
    mod, _ = load_ckp(ckp_path, mod)
    print("Predicting...")
    aucs = []
    accs = []
    for xb, yb in tqdm(data.get_test_data()):
        ypredb = mod(xb).flatten().cpu().detach().numpy()
        yb = yb.float().cpu().detach().numpy()
        aucs.append(auc(yb, ypredb))
        accs.append(acc(yb, ypredb))
    return np.mean(aucs), np.mean(accs)

