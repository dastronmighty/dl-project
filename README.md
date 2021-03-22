# dl-project

[Kaggle Dataset](https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k)

[Kaggle Project Runs](https://www.kaggle.com/fueledbysciened/effnetb0exptfinal)

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

also you can generate statistic images from logs using
```bash
python generate_final_testing_figures.py
```

Furthermore you can qucikly check what the Test AUC score and Test Accuracy Score using the test file. simply
replace the test function, checkpoint path and then run
```bash
python test.py
```

The following simply turns the data into a collage to see what the data looks like.
```bash
python generate_data_figures.py
```