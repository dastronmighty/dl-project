from matplotlib import pyplot as plt

from src.Data.Data import Data
from src.utils.Checkpoint import load_ckp
from src.utils.datautils import sample_from_data_loader
from src.experiments.utils import get_resize_wrapper
from src.models.EfficientNet import EfficientNetB0
from src.utils.utils import tensorToLabels

import torch
import numpy as np

ckp_path = "./checkpoints/FinalEfficientNetB0Checkpoints/FINALEfficientNetB0AUG_EXPT_0_0001_64_Adam_BCELoss/EfficientNetB0AUG_EXPT_0_0001_64_Adam_BCELoss_FINAL.pt"
data_path = "./augmented"


def main():
    seed = 42
    size = 224
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = get_resize_wrapper(size)
    print("Loading Data...")
    data = Data(data_path,
                augmented=True,
                workers=0,
                device=dev,
                total_amt=65_536,
                val_percent=0.1,
                test_amt=1000,
                batch_size=128,
                wrapped_function=wrapper)
    x, y = sample_from_data_loader(data.get_test_data())
    print("Loading Model...")
    mod = EfficientNetB0(classes=1)
    mod, _ = load_ckp(ckp_path, mod)
    print("Predicting...")
    preds = mod(x)
    predictions = ((preds > 0.5) * 1)
    predictions = predictions.flatten()
    peqy = ((y == predictions) * 1)
    idx = []
    for i in range((len(peqy) - 9)):
        idx.append(peqy[i:i + 9].sum().item())
    fidx = np.argmax(idx)
    x, y = x[fidx:fidx + 9], y[fidx:fidx + 9]
    predictions = predictions[fidx:fidx + 9].flatten()
    print("Plotting...")
    rows, cols = 3, 3
    f, axs = plt.subplots(rows, cols, figsize=((5 * cols), (5 * rows)))
    pred_labels, true_labels = tensorToLabels(predictions), tensorToLabels(y)
    for i, x in enumerate(x):
        im = (x.permute(1, 2, 0)).int()
        idx1, idx2 = i // cols, i % cols
        axs[idx1, idx2].imshow(im)
        axs[idx1, idx2].set_title(f"Predicted {pred_labels[i]} - Actual {true_labels[i]}")
        axs[idx1, idx2].axis('off')
    f.savefig(f"./figs/predictions")
    print("Done!")


if __name__ == '__main__':
    main()

