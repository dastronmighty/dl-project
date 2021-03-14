from src.models.BasicCNN import BasicCNN
from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG11, VGG16
from src.models.ResNet import ResNet50, ResNet101, ResNet152
from src.models.EfficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

from src.experiments.utils import summarise_model


def main():
    models = [BasicCNN, BatchNormCNN, VGG11, VGG16,
              ResNet50, ResNet101, ResNet152, EfficientNetB0,
              EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
              EfficientNetB5, EfficientNetB6, EfficientNetB7]
    sizes = [(1, 512, 512), (3, 256, 256), (3, 244, 244), (3, 244, 244),
             (3, 244, 244), (3, 256, 256), (3, 256, 256), (3, 224, 224),
             (3, 240, 240), (3, 260, 260), (3, 300, 300), (3, 380, 380),
             (3, 457, 457), (3, 528, 528), (3, 600, 600)]
    names = ["BasicCNN", "BatchNormCNN", "VGG11", "VGG16",
             "ResNet50", "ResNet101", "ResNet152", "EfficientNetB0",
             "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
             "EfficientNetB5", "EfficientNetB6", "EfficientNetB7"]
    for m, s, n in zip(models, sizes, names):
        with open(f"{n}.txt", "w") as f:
            f.write(n)
            f.write(summarise_model(m, s))
            f.write("\n")


if __name__ == '__main__':
    main()
