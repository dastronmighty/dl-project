from src.experiments.utils import test_model_checkpoint

from src.models.BasicCNN import BasicCNN
from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG11, VGG16
from src.models.ResNet import ResNet50, ResNet101, ResNet152
from src.models.EfficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7


def BasicCNNTest(ckp_path, data_path):
    model_kwargs = {}
    auc, acc = test_model_checkpoint(ckp_path, BasicCNN, model_kwargs, 512, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")

def BatchNormCNNTest(ckp_path, data_path):
    model_kwargs = {}
    auc, acc = test_model_checkpoint(ckp_path, BatchNormCNN, model_kwargs, 256, data_path)


def VGG11Test(ckp_path, data_path):
    model_kwargs = {"in_channels" : 3, "output_size" : 1}
    auc, acc = test_model_checkpoint(ckp_path, VGG11, model_kwargs, 244, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def VGG16Test(ckp_path, data_path):
    model_kwargs = {"in_channels":3, "output_size":1}
    auc, acc = test_model_checkpoint(ckp_path, VGG16, model_kwargs, 244, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def ResNet50Test(ckp_path, data_path):
    model_kwargs = {"in_channels": 3, "classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, ResNet50, model_kwargs, 244, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def ResNet101Test(ckp_path, data_path):
    model_kwargs = {"in_channels": 3, "classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, ResNet101, model_kwargs, 244, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def ResNet152Test(ckp_path, data_path):
    model_kwargs = {"in_channels": 3, "classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, ResNet152, model_kwargs, 244, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB0Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB0, model_kwargs, 244, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB1Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB1, model_kwargs, 240, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB2Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB2, model_kwargs, 260, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB3Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB3, model_kwargs, 300, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB4Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB4, model_kwargs, 380, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB5Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB5, model_kwargs, 457, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB6Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB6, model_kwargs, 528, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")


def EfficientNetB7Test(ckp_path, data_path):
    model_kwargs = {"classes": 1}
    auc, acc = test_model_checkpoint(ckp_path, EfficientNetB7, model_kwargs, 600, data_path)
    print(f"Model from {ckp_path}")
    print(f"auc = {auc}, acc = {acc}")

