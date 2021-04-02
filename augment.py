from src.Data.Augmenter import OcularAugmenter

"""
augment data

DATA_DIR SHOULD BE THE DIRECTORY CREATED BY THE Preprocess program
"""

DATA_DIR = "./preprocessed_data_images"
OUT_DIR = "./"


def main():
    OcularAugmenter(DATA_DIR, OUT_DIR, 256, invert=True, equalize=True, autocontrast=True, verbose=True)


if __name__ == '__main__':
    main()a
