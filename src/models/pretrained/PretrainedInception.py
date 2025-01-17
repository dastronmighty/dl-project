from torch import nn
import torchvision.models as models
import torch


class PretrainedInceptionV3(nn.Module):

    def __init__(self, classes=1, mode="feature"):
        super(PretrainedInceptionV3, self).__init__()
        if mode not in ["feature", "finetuning"]:
            raise RuntimeError("mode must be 'feature' or 'finetuning' ")
        self.model = models.inception_v3(pretrained=True)
        if mode == "feature":
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = torch.nn.Sequential(
            nn.Linear(2048, classes, bias=True),
            nn.Sigmoid())
        self.training = True

    def forward(self, xb):
        xb = self.model(xb)
        if self.training:
            xb = xb.logits
        return xb

    def eval(self):
        self.training = False
        self.model.eval()

    def train(self):
        self.training = True
        self.model.train()
