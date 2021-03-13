from src.experiments.RunExperiment import RunExpt
from src.Data.Data import Data

from src.models.BasicCNN import BasicCNN
from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG11
from src.models.ResNet import ResNet50
from src.models.EfficientNet import EfficientNetB0

directories = {
    "data": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/augmented",
    "log": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints",
    "ckp": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"
}

data = Data(directories["data"], verbose=True)

batch = next(iter(data.get_train_data()))