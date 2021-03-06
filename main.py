from src.models.BasicCNN import Basic_CNN
from src.utils.ParamTuner import ParamTuner

from src.utils.Metrics import auc, acc, highest_tpr_thresh, lowest_fpr_thresh

import torch

DATA_PATH = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"
CKP_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints"
LOG_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"

def main():
    NAME = f"BASIC_CNN_TEST1"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    WORKERS = 4
    SAVE_EVERY = 10
    metrics_dict = {
        "auc": auc,
        "acc": acc,
        "top_tpr_thresh": highest_tpr_thresh,
        "low_fpr_thresh": lowest_fpr_thresh
    }

    lrs = [0.001, 0.0025, 0.008,
           0.0001, 0.00025, 0.0008,
           0.00001, 0.000025, 0.00008,
           0.000001, 0.0000025, 0.000008]
    bss = [32, 64, 128]
    opts = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.Adam]
    lsss = [torch.nn.BCELoss, torch.nn.MSELoss]
    ParamTuner(NAME,
                   Basic_CNN,
                   metrics_dict,
                   "auc",
                   DATA_PATH,
                   LOG_DIR,
                   CKP_DIR,
                   test_amt=0.15,
                   val_amt=0.15,
                   learning_rates=lrs,
                   batch_sizes=bss,
                   optimisers=opts,
                   losses=lsss,
                   EPOCHS=EPOCHS,
                   WORKERS=WORKERS,
                   SAVE_EVERY=SAVE_EVERY,
                   DEVICE=DEVICE,
                   verbose=True)


if __name__ == '__main__':
    main()