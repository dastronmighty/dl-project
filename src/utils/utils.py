from matplotlib import pyplot as plt
from datetime import datetime

def curr_time():
    ct = datetime.now().strftime("%Y%m%d%H%M%S")
    return ct

def tensorToLabels(Y):
    labels = ["Disease", "Normal"]
    to_lab = []
    for y in Y:
        to_lab.append(labels[y])
    return to_lab

def show_dataset(X, Y):
    to_show, l = X[0:9], tensorToLabels(Y[0:9])
    rows, cols = 3, 3
    f, axs = plt.subplots(rows, cols, figsize=((5 * cols), (5 * rows)))
    for i, x in enumerate(to_show):
        im = ((x.permute(1, 2, 0))*255).int()
        idx1, idx2 = i // cols, i % cols
        axs[idx1, idx2].imshow(im)
        axs[idx1, idx2].set_title(i)
        axs[idx1, idx2].axis('off')
    return f
