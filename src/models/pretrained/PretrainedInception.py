from torch import nn
import torchvision.models as models

class PretrainedInceptionV3(nn.Module):

    def __init__(self, classes=1):
        super(PretrainedInceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
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