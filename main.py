from src.models.BasicCNN import Basic_CNN
from src.utils.fitmodel import FitModel
from src.utils.utils import checkpoints_

import torch

PATH = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"
OUT_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints"

def main():
    conf = {
        "name": "basic_cnn",
        "data_path": PATH,
        "dev": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "batch_size": 64,
        "num_workers": 0,
        "epochs": 20,
        "epoch_ckp": 2,
        "lr": 0.008,
        "overwrite": True
    }

    print(f"Configuration:\n{conf}")

    ckp_path = checkpoints_(OUT_DIR, conf)
    conf["ckp_path"] = ckp_path

    model = Basic_CNN()
    Loss = torch.nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=conf["lr"])
    fitted = FitModel(conf, model, opt, Loss, verbose=True)


if __name__ == '__main__':
    main()
