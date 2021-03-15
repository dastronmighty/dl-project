from src.experiments.experiments import BasicCNNExpt

directories = {
    "data": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/augmented",
    "log": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints",
    "ckp": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"
}

aug_directories = {
    "data": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images",
    "log": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/checkpoints",
    "ckp": "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/logs"
}


def main():
    BasicCNNExpt(aug_directories, 0, False)


if __name__ == '__main__':
    main()
