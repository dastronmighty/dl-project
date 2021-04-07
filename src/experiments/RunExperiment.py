import torch

from src.utils.Metrics import auc, acc, highest_tpr_thresh, lowest_fpr_thresh
from src.utils.ParamTuner import ParamTuner, init_folder


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

    log_dir = init_folder(f"{expt_name}_params", directories["log"], True)
    ckp_dir = init_folder(f"{expt_name}_params", directories["ckp"], True)

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

    torch.cuda.empty_cache()
