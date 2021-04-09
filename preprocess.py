from src.Data.Preprocessor import Preproccessor

"""
Preprocess the data

DATA_PATH should be the directory downloaded from kaggle
"""

# path to data might change if you put it somewhere else
DATA_PATH = "./archive"

def main():
    Preproccessor(DATA_PATH, ".", 0, 8000, verbose=True)

if __name__ == '__main__':
    main()
