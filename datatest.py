from src.Data.Data import Data
from torchvision import transforms

DATA_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"

def resize_wrapper(x, y):
    x = transforms.functional.resize(x, size=(244, 244))
    return x, y

data = Data(DATA_DIR,
            augmented=False,
            wrapped_function=resize_wrapper,
            workers=0,
            verbose=True)



DATA_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/augmented"

def resize_wrapper(x, y):
    x = transforms.functional.resize(x, size=(244, 244))
    return x, y

data = Data(DATA_DIR,
            augmented=True,
            wrapped_function=resize_wrapper,
            workers=0,
            verbose=True)
