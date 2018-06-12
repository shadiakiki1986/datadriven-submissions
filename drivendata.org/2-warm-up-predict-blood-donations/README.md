2-warm-up-predict-blood-donations
==============================

solution to drivendata.org competition 2

Competition 2: Warm up: Predict blood donations

Started following the data science cookiecutter templates [here](http://drivendata.github.io/cookiecutter-data-science/)
```
cd 2-warm-up-predict-blood-donations
pew new -r requirements.txt --python=python DRIVENDATA_SUBMISSIONS_2
make
```


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
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
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
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>





Usage
-----

Creation of project

    cd drivendata-submissions
    pew new DRIVEN_2
    pip install cookiecutter
    cookiecutter https://github.com/drivendata/cookiecutter-data-science
    # type in name of folder: 2-warmup-...


Usage of project

    make # display help
    make sync_data_from_s3
    make requirements
    make jupyter


Results
-------
In the end, the ground truth data turned out to already be published, and someone found it and published to get top results on the leaderboard. He was fair though by publishing it on the forum too.

The results here were not so great, and my rank was horrible in the 1000's, but that's only because everyone was in the same ballpark. When I felt there was not much to learn from this, I just stopped and moved on.
