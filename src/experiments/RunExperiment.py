import torch

from src.utils.Metrics import auc, acc, highest_tpr_thresh, lowest_fpr_thresh
from src.utils.ParamTuner import ParamTuner


def RunExpt(expt_name, model, model_kwargs, epochs, directories, wrapper=None):
    torch.cuda.empty_cache()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using Device {DEVICE}")

    EPOCHS = epochs
    WORKERS = 2
    SAVE_EVERY = 10

    metrics_dict = {
        "auc": auc,
        "acc": acc,
        "top_tpr_thresh": highest_tpr_thresh,
        "low_fpr_thresh": lowest_fpr_thresh
    }

    lrs = [0.003]
    bss = [64]
    opts = [torch.optim.Adam]
    losses = [torch.nn.BCELoss]

    tune = ParamTuner(expt_name,
                      model,
                      metrics_dict,
                      "auc",
                      directories["data"],
                      directories["log"],
                      directories["ckp"],
                      model_kwargs=model_kwargs,
                      total_amt=16000,
                      val_percent=0.2,
                      test_amt=1000,
                      learning_rates=lrs,
                      batch_sizes=bss,
                      optimisers=opts,
                      losses=losses,
                      EPOCHS=EPOCHS,
                      wrapped_function=wrapper,
                      WORKERS=WORKERS,
                      SAVE_EVERY=SAVE_EVERY,
                      DEVICE=DEVICE,
                      verbose=True,
                      overwrite=True)

    print(f"EXPERIMENT {expt_name} FINISHED")
    print(f"OUTPUT: {tune.trials}")
    torch.cuda.empty_cache()
