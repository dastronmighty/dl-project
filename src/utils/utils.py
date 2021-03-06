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
    f, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i, x in enumerate(to_show):
        im = ((x.permute(1, 2, 0))*255).int()
        axs[i//3, i%3].imshow(im)
        axs[i//3, i%3].set_title(l[i])
        axs[i//3, i%3].axis('off')
    return f

