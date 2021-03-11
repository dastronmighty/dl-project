from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG_net
from src.utils.ParamTuner import ParamTuner

from src.utils.datautils import mount_to_device

from src.utils.Metrics import auc, acc, highest_tpr_thresh, lowest_fpr_thresh

import torch
from torchvision import transforms

DATA_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/augmented"
CKP_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints"
LOG_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"


def resize_wrapper(x, y):
    x = transforms.functional.resize(x, size=(244, 244))
    return x, y


def main():
    NAME = f"VGGTEST"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device {DEVICE}")
    EPOCHS = 15
    WORKERS = 0
    SAVE_EVERY = 10
    metrics_dict = {
        "auc": auc,
        "acc": acc,
        "top_tpr_thresh": highest_tpr_thresh,
        "low_fpr_thresh": lowest_fpr_thresh
    }

    lrs = [0.00025]
    bss = [32]
    opts = [torch.optim.Adam]
    losses = [torch.nn.BCELoss]

    ParamTuner(NAME,
                VGG_net,
                metrics_dict,
                "auc",
                DATA_DIR,
                LOG_DIR,
                CKP_DIR,
                val_percent=0.2,
                test_amt=3000,
                learning_rates=lrs,
                batch_sizes=bss,
                optimisers=opts,
                losses=losses,
                EPOCHS=EPOCHS,
                WORKERS=WORKERS,
                wrapped_function=resize_wrapper,
                SAVE_EVERY=SAVE_EVERY,
                DEVICE=DEVICE,
                verbose=True,
                overwrite=True)


if __name__ == '__main__':
    main()
