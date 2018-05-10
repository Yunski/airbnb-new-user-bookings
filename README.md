# airbnb-new-user-bookings
Predict which country a new user will make his or her first booking. \
Link to the kaggle challenge: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

# Getting Started
Install [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html).
Then create the environment with conda.
```
$ conda env create -f environment.yml
```
### Download data
Download the Airbnb data [here](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data).
### Create data and results directory
i.e. 
```
$ mkdir data results
```
### Generating train and test files
Run the following script to generate the processed train and test files.
```
$ python preprocess.py
```
Use the -d flag to change your data directory (default: 'data').
### Training Models
```
$ python train.py -h
usage: train.py [-h] [-d DATA_DIR] [-s RESULTS_DIR]
                [-m {all,logistic,tree,forest,ada,xgb}]
                [-o {random,smote,adasyn,none}] [-k K_FOLDS]
                [--device {cpu,gpu}]

Airbnb New User Booking Classification

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR           data directory
  -s RESULTS_DIR        results save directory
  -m {all,logistic,tree,forest,ada,xgb}
                        model
  -o {random,smote,adasyn,none}
                        oversampling method
  -k K_FOLDS            number of CV folds
  --device {cpu,gpu}    device
```
To use lightgbm, run:
```
$ python lightgbm.py
```
Note that gpu is not faster for such a small dataset.

### Results
See cross validation output in the ```results``` directory in the form of pickle files.
