import numpy as np
import pandas as pd 
import argparse 
import pickle
import time
import os
import lightgbm as lgb

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import KFold 

from utils import get_train, evaluate

def train(oversampling_method, k_folds, data_dir, results_dir, device='cpu', verbose=True):
    start_time = time.time()
    if verbose:
        print("Using device: {}".format(device))
        print("Reading train data in...")
    X_train, Y_train, feature_labels = get_train(data_dir, one_hot=False)
    if verbose:
        print("Successfully loaded data.")
    print("Starting Cross-Validation with {} folds...".format(k_folds)) 
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(X_train)
    categories = ['age_bucket', 'gender', 'signup_method', 'signup_flow', 
                  'language', 'affiliate_channel', 'affiliate_provider', 
                  'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    params = {
        'task': 'train',
        'objective': 'multiclass',
        'num_class': 12,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': int(verbose),
        'device': device,
        'gpu_device_id': 0,
        'gpu_platform_id': 0,
        'max_bin': 63
    }
    for k, (train_index, test_index) in enumerate(kf.split(X_train)):
        print("Processing Fold {} out of {}".format(k+1, k_folds))
        X_trainCV, X_testCV = X_train[train_index], X_train[test_index]
        Y_trainCV, Y_testCV = Y_train[train_index], Y_train[test_index]
        if verbose:
            print("{} oversampling process started...".format(oversampling_method))
        curr_time = time.time()
        if oversampling_method == "adasyn":
            X_train_resampled, Y_train_resampled = ADASYN().fit_sample(X_trainCV, Y_trainCV)
        elif oversampling_method == "smote":
            X_train_resampled, Y_train_resampled = SMOTE().fit_sample(X_trainCV, Y_trainCV)
        elif oversampling_method == "random":
            X_train_resampled, Y_train_resampled = RandomOverSampler(random_state=0).fit_sample(X_trainCV, Y_trainCV)
        else:
            X_train_resampled, Y_train_resampled = X_trainCV, Y_trainCV
        if verbose:
            print("Oversampling completed")
            print("Time Taken: {:.2f}".format(time.time()-curr_time))
            print("Size of Oversampled data: {}".format(X_train_resampled.shape))
        curr_time = time.time() 
        lgb_train = lgb.Dataset(data=X_train_resampled, label=Y_train_resampled, feature_name=feature_labels)
        gbm = lgb.train(params, lgb_train, num_boost_round=20)
        print("Time taken: {:.2f}".format(time.time()-curr_time))
        Y_probs = gbm.predict(X_testCV)
        print(Y_probs[:10])
        result = evaluate(Y_testCV, Y_probs)
        pickle.dump(result, open(os.path.join(results_dir, "lgbm_fold_{}.p".format(k+1)), "wb" ))

    print("Training took {:.2f}s.".format(time.time()-start_time))
    print("Finished.") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb New User Booking Classification")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-s', help='results save directory', dest='results_dir', type=str, default="results")
    parser.add_argument('-o', help='oversampling method', dest='oversampling_method', type=str, default="smote", 
                        choices=["random", "smote", "adasyn", "none"])
    parser.add_argument('-k', help='number of CV folds', dest='k_folds', type=int, default=5)
    parser.add_argument('--device', help='device', dest='device', type=str, default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    train(args.oversampling_method, args.k_folds, args.data_dir, args.results_dir, args.device, verbose=True)
    
