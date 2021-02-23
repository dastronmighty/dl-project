from src.Data.Preprocessor import Preproccessor

DATA_PATH = "../data.nosync"

def main():
    Preproccessor(DATA_PATH, 0, 8000, verbose=True)

if __name__ == '__main__':
    main()
