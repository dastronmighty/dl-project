import os
import torch
import shutil
from src.utils.utils import curr_time, init_folder


class Checkpoint:

    def __init__(self, name, PATH, save_every, overwrite=False):
        self.name = name
        self.overwrite = overwrite
        self.path = self.init_folder(name, PATH, overwrite)
        self.save_every = save_every

    def save(self, epoch, model, opt=None, add_tag=None):
        if epoch % self.save_every == 0 and epoch != 0:
            self.save_helper(epoch, model, opt=opt, add_tag=add_tag)

    def save_override(self, epoch, model, opt=None, add_tag=None):
        self.save(epoch, model, opt=opt, add_tag=add_tag)

    def save_helper(self, epoch, model, opt=None, add_tag=None):
        ct = curr_time()
        p = f"{self.path}/{self.name}_{ct}_{epoch}.pt"
        if add_tag is not None:
            p = f"{self.path}/{self.name}_{add_tag}.pt"
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        if opt is not None:
            checkpoint['optimizer_state_dict'] = opt.state_dict()
        torch.save(checkpoint, p)


def load_ckp(checkpoint_path, model, opt=None, dev="cpu"):
    # load check point
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])
    # initialize optimizer from checkpoint to optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # return model, optimizer
    return model, opt


def load_ckp(checkpoint_path, model, opt=None, dev="cpu"):
    # load check point
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])
    # initialize optimizer from checkpoint to optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # return model, optimizer
    return model, opt
