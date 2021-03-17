import torch
from src.utils.utils import curr_time, init_folder


class Checkpoint:

    def __init__(self, name, PATH, save_every, overwrite=False):
        """
        A Class for creating checkpoints for models
        :param name: the name of the current experiment
        :param PATH: the path to save to
        :param save_every: how often (per epoch) to save
        :param overwrite: whether to overwrite other checkpoints with the same name
        """
        self.name = name
        self.overwrite = overwrite
        self.path = init_folder(name, PATH, overwrite)
        self.save_every = save_every

    def save(self, epoch, model, opt=None, add_tag=None):
        """
        The save function
        :param epoch: the current epoch
        :param model: current model
        :param opt: current optimizer
        :param add_tag: an optional tag
        """
        if epoch % self.save_every == 0 and epoch != 0:
            self.save_helper(epoch, model, opt=opt, add_tag=add_tag)

    def save_override(self, epoch, model, opt=None, add_tag=None):
        """
        A funciton to save without checking the epoch
        :param epoch: the current epoch
        :param model: current model
        :param opt: current optimizer
        :param add_tag: an optional tag
        """
        self.save_helper(epoch, model, opt=opt, add_tag=add_tag)

    def save_helper(self, epoch, model, opt=None, add_tag=None):
        """
        The function that actually saves the model and optimizers
        :param epoch: the current epoch
        :param model: current model
        :param opt: current optimizer
        :param add_tag: an optional tag
        """
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
    """
    load a model checkpoint
    :param checkpoint_path: the path to the .pt file
    :param model: an initalised version of the same class as the model being loaded
    :param opt: an initalised version of the same optimiser being loaded
    :param dev: what device to use (note it must match what device was used when saved or sometimes can bug out)
    :return: the loaded model and optimizer
    """
    # load check point
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])
    # initialize optimizer from checkpoint to optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # return model, optimizer
    return model, opt

