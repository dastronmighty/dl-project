from torch import nn


class VGG_net(nn.Module):

    def __init__(self, in_channels=3, output_size=1, architecture=11):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        VGG11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

        arch = VGG11
        if architecture == 16:
            arch = VGG16
        self.conv_layers = self.create_conv_layers(arch)

        self.flat = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_features=output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        xb = self.conv_layers(xb)
        xb = self.flat(xb)
        xb = self.dense(xb)
        return xb

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for layer_type in architecture:
            if type(layer_type) == int:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=layer_type, kernel_size=3, padding=1)]
                layers += [nn.BatchNorm2d(layer_type)]
                layers += [nn.ReLU()]
                in_channels = layer_type
            elif layer_type == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif layer_type == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)