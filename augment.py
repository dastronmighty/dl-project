from src.Data.Augmenter import OcularAugmenter

DATA_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"
OUT_DIR = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1"

def main():
    OcularAugmenter(DATA_DIR, OUT_DIR, 256, invert=True, equalize=True, autocontrast=True, verbose=True)

if __name__ == '__main__':
    main()
