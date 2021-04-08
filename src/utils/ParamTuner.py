from src.Data.Data import Data
from src.utils.Logger import Logger
from src.utils.Checkpoint import Checkpoint
from src.utils.fitmodel import FitModel
from src.experiments.utils import test_ckps

import random
import numpy as np

import torch

class ParamTuner:

    def __init__(self,
                 Name,
                 model_class,
                 metrics_to_use,
                 metric_to_optimise,
                 DATA_DIR,
                 LOG_DIR,
                 CKP_DIR,
                 augmented,
                 model_kwargs,
                 total_amt=3000,
                 val_percent=0.2,
                 test_amt=3000,
                 learning_rates=[0.001],
                 batch_sizes=[32],
                 optimisers=[torch.optim.SGD],
                 losses=[torch.nn.BCELoss],
                 SAVE_EVERY=2,
                 EPOCHS=20,
                 DEVICE='cpu',
                 wrapped_function=None,
                 WORKERS=0,
                 verbose=False,
                 overwrite=False,
                 train_early_stopping=True,
                 test_early_stopping=True,
                 early_stopping_attention=4,
                 seed=42):
        """
        This was a class made for the purposes of hyperparameter tuning to wrap everything together
        :param Name: The name of the experiment
        :param model_class: the class of the model to use for tuning
        :param metrics_to_use: the metrics to use
        :param metric_to_optimise: the metric we are looking to optimise
        :param DATA_DIR: the data directory path
        :param LOG_DIR: the directory to save logs to
        :param CKP_DIR: the directory to save checkpoints to
        :param augmented: whether this experiment is using augmented data or not
        :param model_kwargs: the arguments to pass to the model initialise
        :param total_amt: the amount of data to use for training
        :param val_percent: the percent of total data to use for validation
        :param test_amt: the amount of data to set aside for testing after tuning
        :param learning_rates: a list of learning rates to try length >= 1
        :param batch_sizes: a list of batch sizes to try length >= 1
        :param optimisers: a list of optimizers to try length >= 1
        :param losses: a list of loss functions to try length >= 1
        :param SAVE_EVERY: how often to save the models and optimizers
        :param EPOCHS: the number of epochs to train for
        :param DEVICE: the device to use
        :param wrapped_function: a wrapped function for data loading if needed
        :param WORKERS: the number of workers for dataloading to use
        :param verbose: whether to print the status of whats happening
        :param overwrite: whether to overwrite previous experiments with the same name
        :param seed: a seed for reproducibility
        """

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.augmented = augmented

        self.seed = seed

        self.name = Name
        self.model_class = model_class
        self.epochs = EPOCHS
        self.device = DEVICE
        self.workers = WORKERS
        self.wrapped_function = wrapped_function
        self.verbose = verbose

        self.overwrite = overwrite
        self.save_every = SAVE_EVERY

        self.metrics_to_use = metrics_to_use
        self.metric_to_optimise = metric_to_optimise

        self.total_amt = total_amt
        self.test_amt = test_amt
        self.val_percent = val_percent

        self.DATA_DIR = DATA_DIR
        self.LOG_DIR = LOG_DIR
        self.CKP_DIR = CKP_DIR

        self.model_kwargs = model_kwargs

        self.tres = train_early_stopping
        self.tes = test_early_stopping
        self.es_attn = early_stopping_attention

        self.trials = {}
        for lr in learning_rates:
            for bs in batch_sizes:
                for optim in optimisers:
                    for lf in losses:
                        n, m = self.run_trial(lr, bs, optim, lf)
                        self.trials[n] = m
                        if self.verbose:
                            print(self.trials)

        loss_func = losses[0]()
        test_ckps(data_dir=self.DATA_DIR,
              auged=self.augmented,
              ckp_dir=self.CKP_DIR,
              log_dir=self.LOG_DIR,
              model=self.model_class,
              model_kwargs=self.model_kwargs,
              mets=metrics_to_use,
              device=self.device,
              loss_func=loss_func,
              total_amt=self.total_amt,
              val_percent=self.val_percent,
              test_amt=self.test_amt,
              wrapped_function=wrapped_function,
              workers=self.workers,
              seed=self.seed)

    def run_trial(self, LR, BATCH_SIZE, OPTIM, LOSS):
        """
        Initalise all over again for training on given Learning rate, Batch size, Optimizer and loss function
        This function basically just initialises everything and sends it to the fitmodel
        :param LR: Learning Rate
        :param BATCH_SIZE: Batch Size
        :param OPTIM: Optimiser
        :param LOSS: Loss Function
        :return: The final metric to optimise score from the training
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        NAME = f"{self.name}_{str(LR).replace('.', '_')}"
        NAME += f"_{BATCH_SIZE}"

        model = self.model_class(**self.model_kwargs)

        opt = OPTIM(model.parameters(), lr=LR)
        opt_str = str(type(opt)).split("'")[-2].split(".")[-1]
        NAME += f"_{opt_str}"

        loss_func = LOSS()
        loss_func_str = str(type(loss_func)).split("'")[-2].split(".")[-1]
        NAME += f"_{loss_func_str}"

        data = Data(self.DATA_DIR,
                    self.augmented,
                    batch_size=BATCH_SIZE,
                    total_amt=self.total_amt,
                    val_percent=self.val_percent,
                    test_amt=self.test_amt,
                    wrapped_function=self.wrapped_function,
                    workers=self.workers,
                    device=self.device,
                    verbose=self.verbose,
                    seed=self.seed)

        logger = Logger(NAME,
                        self.LOG_DIR,
                        self.metrics_to_use,
                        train_early_stopping=self.tres,
                        test_early_stopping=self.tes,
                        stopping_attention=self.es_attn,
                        overwrite=self.overwrite,
                        verbose=self.verbose)

        checkpointer = Checkpoint(NAME,
                            self.CKP_DIR,
                            self.save_every,
                            overwrite=self.overwrite)

        FitModel(model,
                 data,
                 opt,
                 loss_func,
                 self.epochs,
                 self.device,
                 logger,
                 checkpointer,
                 verbose=self.verbose,
                 seed=self.seed)

        met_final = logger.test_history[self.metric_to_optimise][-1]

        return NAME, met_final
