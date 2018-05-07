# airbnb-new-user-bookings
Predict which country a new user will make his or her first booking.

# Getting Started
Install [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html).
Then create the environment with conda.
```
$ conda env create -f environment.yml
```
### Download data
Download the Airbnb data [here](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data).
Run the following script to generate the processed train and test files.
```
$ python data.py
```
Use the -d flag to change your data directory (default: 'data').
