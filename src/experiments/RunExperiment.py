from src.utils.testmodelutils import test_models_on_batch_and_show
from src.utils.Metrics import auc, acc, highest_tpr_thresh, lowest_fpr_thresh
from src.utils.ParamTuner import ParamTuner
from src.utils.utils import init_folder, make_folder_if_not_there
from src.utils.Plotting import make_plot

from src.experiments.utils import get_resize_wrapper, get_pretrained_size_and_norm_wrapper

import torch

import os


def RunExpt(expt_name,
            model,
            model_kwargs,
            epochs,
            directories,
            augmented,
            wrapper=None,
            lrs=[0.003],
            bss=[64],
            opts=[torch.optim.Adam],
            losses=[torch.nn.BCELoss],
            workers=2,
            save_every=2,
            **kwargs):
    """
    Run an experiment
    Basically just bunch all the things I need together for hyperparam tuning so I dont have to always write it out
    also now we can wrap it up for different models aswell
    :param expt_name: the name of the experiment
    :param model:the class of the model to use for tuning
    :param model_kwargs: the arguments to pass to the model initialise
    :param epochs: the number of epochs to train for
    :param directories: a dictionary of the data, log, and checkpoint directories
    :param augmented: whether this experiment is using augmented data or not
    :param wrapper: a wrapper function for data loading if needed
    :param lrs: a list of learning rates to try length >= 1
    :param bss: a list of batch sizes to try length >= 1
    :param opts: a list of optimizers to try length >= 1
    :param losses: a list of loss functions to try length >= 1
    :param workers: the number of workers for dataloading to use
    :param save_every: how often to save the models and optimizers
    :param kwargs: the optional kwargs to pass to the param tuner
    """
    torch.cuda.empty_cache()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using Device {DEVICE}")

    EPOCHS = epochs
    WORKERS = workers

    metrics_dict = {
        "auc": auc,
        "acc": acc,
        "top_tpr_thresh": highest_tpr_thresh,
        "low_fpr_thresh": lowest_fpr_thresh
    }

    log_dir = init_folder(f"{expt_name}", directories["log"], True)
    ckp_dir = init_folder(f"{expt_name}", directories["ckp"], True)

    tune = ParamTuner(expt_name,
                      model,
                      metrics_dict,
                      "auc",
                      directories["data"],
                      log_dir,
                      ckp_dir,
                      augmented,
                      model_kwargs=model_kwargs,
                      learning_rates=lrs,
                      batch_sizes=bss,
                      optimisers=opts,
                      losses=losses,
                      EPOCHS=EPOCHS,
                      wrapped_function=wrapper,
                      WORKERS=WORKERS,
                      SAVE_EVERY=save_every,
                      DEVICE=DEVICE,
                      verbose=True,
                      overwrite=True,
                      **kwargs)

    print(f"HYPER-PARAM OUTPUT: {tune.trials}")
    print(f"EXPERIMENT {expt_name} FINISHED")

    print("PLOTTING!")

    cwd = os.getcwd()
    fig_dir = make_folder_if_not_there("figs", cwd)
    tuned_fig_dir = init_folder(expt_name, fig_dir, True)
    make_plot(log_dir, tuned_fig_dir)

    test_data_path = ""
    if "test" in directories.keys():
        test_data_path = directories["test"]
    else:
        test_data_path = directories["data"]
    test_models_on_batch_and_show(expt_name, test_data_path, ckp_dir, model, model_kwargs, dev=DEVICE, wrapped=wrapper)

    torch.cuda.empty_cache()


def base_experiment(name,
                    model,
                    directories,
                    workers,
                    model_kwargs,
                    aug,
                    wrapper=None,
                    lrs=[0.1, 0.01, 0.001, 0.0001],
                    bss=[32, 64],
                    opts=[torch.optim.Adam],
                    losses=[torch.nn.BCELoss],
                    **kwargs):
    """
    Run a single experiment
    :param name: name of experiment
    :param model: the class of the model to use
    :param directories: a dictionary of the data, logs, and checkpoints
    :param workers: the number of workers to use for data loading
    :param model_kwargs: the arguments to pass to the model when initialized
    :param aug: wether ot not the data is augmented
    :param wrapper: Wrapper required
    :param lrs: The learning rates to try
    :param bss: The Batch Sizes to try
    :param opts: The Optimizers to try
    :param losses: The Loss Functions to try
    :param kwargs: the optional kwargs to pass to the param tuner
    """
    RunExpt(f"{name}_EXPT",
            model,
            model_kwargs,
            100,
            directories,
            aug,
            wrapper,
            lrs=lrs,
            bss=bss,
            opts=opts,
            losses=losses,
            workers=workers,
            save_every=2,
            train_early_stopping=False,
            test_early_stopping=True,
            **kwargs)


def regular_experiment(name,
                       model,
                       directories,
                       size,
                       workers,
                       model_kwargs,
                       aug,
                       lrs=[0.1, 0.01, 0.001, 0.0001],
                       bss=[32, 64],
                       opts=[torch.optim.Adam],
                       losses=[torch.nn.BCELoss],
                       **kwargs):
    """
    Run a single experiment on an untrained model
    :param name: name of experiment
    :param model: the class of the model to use
    :param directories: a dictionary of the data, logs, and checkpoints
    :param size: size to resize samples to
    :param workers: the number of workers to use for data loading
    :param model_kwargs: the arguments to pass to the model when initialized
    :param aug: wether ot not the data is augmented
    :param lrs: The learning rates to try
    :param bss: The Batch Sizes to try
    :param opts: The Optimizers to try
    :param losses: The Loss Functions to try
    :param kwargs: the optional kwargs to pass to the param tuner
    """
    wrapper = get_resize_wrapper(size)
    base_experiment(name, model, directories, workers, model_kwargs, aug, wrapper, lrs=lrs, bss=bss, opts=opts,
                    losses=losses, **kwargs)


def pretrained_experiment(name,
                          model,
                          directories,
                          size,
                          workers,
                          model_kwargs,
                          aug,
                          lrs=[0.1, 0.01, 0.001, 0.0001],
                          bss=[32, 64],
                          opts=[torch.optim.Adam],
                          losses=[torch.nn.BCELoss],
                          **kwargs):
    """
    Run a single experiment on a pretrained model
    :param name: name of experiment
    :param model: the class of the model to use
    :param directories: a dictionary of the data, logs, and checkpoints
    :param size: size to resize samples to
    :param workers: the number of workers to use for data loading
    :param model_kwargs: the arguments to pass to the model when initialized
    :param aug: wether ot not the data is augmented
    :param lrs: The learning rates to try
    :param bss: The Batch Sizes to try
    :param opts: The Optimizers to try
    :param losses: The Loss Functions to try
    :param kwargs: the optional kwargs to pass to the param tuner
    """
    wrapper = get_pretrained_size_and_norm_wrapper(size)
    base_experiment(name, model, directories, workers, model_kwargs, aug, wrapper, lrs=lrs, bss=bss, opts=opts,
                    losses=losses, **kwargs)
