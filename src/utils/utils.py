from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import shutil

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

def get_compact_stats(stats):
    all_stats_dict = stats[0]
    for s_ in stats[1:]:
        for k in s_.keys():
            if k in ["auc", "acc"]:
                all_stats_dict[k] = (all_stats_dict[k] + s_[k])/2
            else:
                all_stats_dict[k] = np.append(all_stats_dict[k], s_[k])
    cstats = {
        "avg_auc": all_stats_dict["auc"],
        "avg_acc": all_stats_dict["acc"],
        "fpr_stats": get_stats_helper(all_stats_dict["fpr"], all_stats_dict["thresholds"]),
        "tpr_stats": get_stats_helper(all_stats_dict["tpr"], all_stats_dict["thresholds"]),
    }
    return cstats

def get_stats_helper(arr, threshholds):
    stats = {}
    s = len(arr)
    max_idx = arr.argmax()
    mean_idx = arr.argsort()[int(s/2)]
    min_idx = arr.argsort()[0]
    stats["mean"] = arr.mean()
    stats["max"] = arr[max_idx]
    stats["max_threshold"] = threshholds[max_idx]
    stats["median"] = arr[mean_idx]
    stats["median_threshold"] = threshholds[mean_idx]
    stats["min"] = arr[min_idx]
    stats["min_theshold"] = threshholds[min_idx]
    return stats

def checkpoints_(path, conf):
    p = f"{path}/checkpoints"
    if conf["epoch_ckp"] is not None:
        if "checkpoints" not in os.listdir(path):
            os.mkdir(p)
            os.mkdir(f"{p}/{conf['name']}")
        else:
            if conf['name'] in os.listdir(p):
                if conf["overwrite"]:
                    shutil.rmtree(f"{p}/{conf['name']}")
                else:
                    warn = f"Folder {conf['name']} already existed saving to "
                    conf["name"] = conf["name"]+"1"
                    warn += conf["name"]
                    print(warn)
                    os.mkdir(f"{p}/{conf['name']}")
    return f"{p}/{conf['name']}"

def save_ckp(model, opt, stats, name, epoch, PATH):
    p =  f"{PATH}/{name}_{epoch}.pt"
    checkpoint = {
            'epoch': epoch,
            'stats': stats,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }
    torch.save(checkpoint, p)

def load_ckp(checkpoint_path, agent, dev):
    # load check point
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    # initialize state_dict from checkpoint to model
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.update_target()
    agent.current_step = checkpoint['steps']
    # initialize optimizer from checkpoint to optimizer
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # initialize valid_loss_min from checkpoint to valid_loss_min

    # return model, optimizer, epoch value, min validation loss
    return agent, checkpoint['episode'], checkpoint["stats"]
