# Hand data science
==============================
Data science for hand shape and orientation
## Run
- Run pipeline for small data but for code any path should be `big/src/stage/*.py` in `big/dvc.yaml`
```bash
dvc repro
``` 
- Run pipeline for small data but for code any path should be `src/stage/*.py` in `dvc.yaml` 
```bash
dvc repro big/dvc.yaml
``` 
- Открыть камеру, показать руку камере и нажать на "Пробел", чтобы детектировать руку и через `1` сек выводить другое окно. `0` - номер камеры
```bash
python3 big/src/apps/word_image.py 0 1
```
- Найти слово в файлах
```bash
grep -i 's14c' folder/*.csv
```
- Определить статистику о метках по *.csv и ввод одна путь для json и несколько путей для csv
```bash
python3 big/src/features/stat_fsw_handshape.py
```


## Project Organization(Примерно)
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── big                <- Source code and data for runing use big data. 
    │   ├── data               <- Big data.    
    │   │   ├── external       <- Data from third party sources.
    │   │   ├── interim        <- Intermediate data that has been transformed.
    │   │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   └── raw            <- The original, immutable data dump.       
    │   └── src                <- Source code for use in this project.
    │       ├── __init__.py    <- Makes src a Python module
    │       │
    │       ├── data           <- Scripts to download or generate data
    │       │   └── make_dataset.py
    │       │
    │       ├── features       <- Scripts to turn raw data into features for modeling
    │       │   └── build_features.py
    │       │
    │       ├── models         <- Scripts to train models and then use trained models to make
    │       │   │                 predictions
    │       │   ├── predict_model.py
    │       │   └── train_model.py
    │       │
    │       └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │           └── visualize.py
    ├── data               <- Small data.    
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
