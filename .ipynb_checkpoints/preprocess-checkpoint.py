from src.Data.Preprocessor import Preproccessor

"""
Preprocess the data

DATA_PATH should be the directory downloaded from kaggle
"""

DATA_PATH = "./data"

def main():
    Preproccessor(DATA_PATH, 0, 8000, verbose=True)

if __name__ == '__main__':
    main()
