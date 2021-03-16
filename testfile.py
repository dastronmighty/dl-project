import os


from src.utils.Metrics import auc, acc, highest_tpr_thresh, lowest_fpr_thresh
from src.Data.Data import Data
from src.experiments.utils import get_resize_wrapper
from src.models.BasicCNN import BasicCNN

import torch

DATA_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"
CKP_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints"
LOG_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"


metrics_dict = {
    "auc": auc,
    "acc": acc,
    "top_tpr_thresh": highest_tpr_thresh,
    "low_fpr_thresh": lowest_fpr_thresh
}

loss_func = torch.nn.BCELoss()

wrapper = get_resize_wrapper(512)




