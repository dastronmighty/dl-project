from src.Data.Data import Data
from src.utils.Logger import Logger
from src.utils.Checkpoint import Checkpoint
from src.utils.fitmodel import FitModel

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
                 test_amt=0.15,
                 val_amt=0.15,
                 learning_rates=[0.001],
                 batch_sizes=[32],
                 optimisers=[torch.optim.SGD],
                 losses=[torch.nn.BCELoss],
                 SAVE_EVERY=10,
                 EPOCHS=20,
                 DEVICE='cpu',
                 wrapped_function=None,
                 WORKERS=0,
                 verbose=False,
                 overwrite=False,
                 seed=42):

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

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

        self.test_amt = test_amt
        self.val_amt = val_amt

        self.DATA_DIR = DATA_DIR
        self.LOG_DIR = LOG_DIR
        self.CKP_DIR = CKP_DIR

        self.trials = {}
        for lr in learning_rates:
            for bs in batch_sizes:
                for optim in optimisers:
                    for lf in losses:
                        n, m = self.run_trial(lr, bs, optim, lf)
                        self.trials[n] = m
                        if self.verbose:
                            print(self.trials)
        print(self.trials)

    def run_trial(self, LR, BATCH_SIZE, OPTIM, LOSS):
        NAME = f"{self.name}_{str(LR).replace('.', '_')}"
        NAME += f"_{BATCH_SIZE}"

        model = self.model_class()

        opt = OPTIM(model.parameters(), lr=LR)
        opt_str = str(type(opt)).split("'")[-2].split(".")[-1]
        NAME += f"_{opt_str}"

        loss_func = LOSS()
        loss_func_str = str(type(loss_func)).split("'")[-2].split(".")[-1]
        NAME += f"_{loss_func_str}"

        data = Data(self.DATA_DIR,
                    batch_size=BATCH_SIZE,
                    val_amt=self.val_amt,
                    test_amt=self.test_amt,
                    wrapped_function=self.wrapped_function,
                    workers=self.workers,
                    device=self.device,
                    seed=self.seed)

        logger = Logger(NAME,
                        self.LOG_DIR,
                        self.metrics_to_use,
                        train_early_stopping=True,
                        test_early_stopping=True,
                        stopping_attention=4,
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
