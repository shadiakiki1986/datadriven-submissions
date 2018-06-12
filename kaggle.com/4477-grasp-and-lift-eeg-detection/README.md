4477-grasp-and-lift-eeg-detection
=================================

solution to kaggle.com competition

Identify hand motions from EEG recordings

https://www.kaggle.com/c/grasp-and-lift-eeg-detection/

This competition was selected from https://github.com/dmlc/xgboost/tree/master/demo
because it was a regression from time series features

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

    make create_environment
    pip install numpy Cython # needed for auto-sklearn in requirements.txt
    make requirements

    # get kaggle API token as described https://github.com/Kaggle/kaggle-api#api-credentials
    # accept competition rules from https://www.kaggle.com/c/grasp-and-lift-eeg-detection/rules
    # make sure there is at least 5 GB of space
    make sync_data_from_s3 # requires kaggle api token above

Subsequent usage

    pew workon 4477-grasp-and-lift-eeg-detection
    sudo pew in 4477-grasp-and-lift-eeg-detection make jupyter # sudo for https


Results
-------
I'm going to apply the network architecture that best worked with drivendata.org's competition 44-dengai.
It's the architecture with LSTM-based AE on both features and targets along with a middle fully connected feedforward layer.
