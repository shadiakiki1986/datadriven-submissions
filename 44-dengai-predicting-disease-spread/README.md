44-dengai-predicting-disease-spread
===================================

solution to drivendata.org competition 44

predicting disease spread

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>





Usage
-----

Creation of project

    cd drivendata-submissions
    pew new DRIVEN_1
    pip install cookiecutter
    cookiecutter https://github.com/drivendata/cookiecutter-data-science
    # type in name of folder: 1-united-...


First usage of project

    make # display help
    make sync_data_from_s3
    make create_environment
    pip install numpy Cython # needed for auto-sklearn in requirements.txt
    make requirements

Subsequent usage

    pew workon 1-united-nations-millennium-development-goals
    make jupyter


Results
-------
Using ML RF didn't get me anywhere in this.

I tried to reshape the data similar to my previous project with http://gsquaredcapital.com/ but it didn't improve the published result on the leaderboard.

The most successful model was simply an auto-regression using only the features that are to be predicted. Using the set `(t-1, t-2, t-3, ...)` resulted in better scores than just `(t-1)`.

I'm not sure how I could have applied ML here.

Feature reduction was unstable because the time dimension is only 36 samples long (36 years).
