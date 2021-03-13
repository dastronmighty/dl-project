from src.experiments.RunExperiment import RunExpt
from src.utils.utils import get_resize_wrapper

from src.models.VGG_net import VGG11

directories = {
    "data": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/augmented",
    "log": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints",
    "ckp": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"
}

def main():
    wrapper = get_resize_wrapper(244)
    model = VGG11
    model_kwargs = {"in_channels": 3, "output_size": 1}
    RunExpt("VGG11TEST", model, model_kwargs, 15, directories, wrapper)

if __name__ == '__main__':
    main()