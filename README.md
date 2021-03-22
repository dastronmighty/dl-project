# dl-project

[Kaggle Dataset](https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k)

[Kaggle Project](https://www.kaggle.com/fueledbysciened/effnetb0exptfinal)

### How to use this project

```bash
pip install -r requirements.txt
```

then download the kaggle data to the root directory and name the folder ```data```

to get the required preprocessed images run
```bash
python preprocess.py
```

To optionally get augmented data run

```bash
python augment.py
```

You can now run any experiment from src.experiments.experiements
