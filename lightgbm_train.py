import numpy as np
import pandas as pd 
import argparse 
import pickle
import time
import os
import lightgbm as lgb

from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import KFold 

from utils import get_train, evaluate

def train(sampling_method, k_folds, data_dir, results_dir, device='cpu', verbose=True):
    start_time = time.time()
    if verbose:
        print("Using device: {}".format(device))
        print("Reading train data in...")
    X_train, Y_train, feature_labels = get_train(data_dir)
    if verbose:
        print("Successfully loaded data.")
    print("Starting Cross-Validation with {} folds...".format(k_folds)) 
    kf = KFold(n_splits=k_folds, shuffle=True)
    kf.get_n_splits(X_train)
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
            print("{} sampling process started...".format(sampling_method))
        curr_time = time.time()
        if sampling_method == "adasyn":
            X_train_resampled, Y_train_resampled = ADASYN().fit_sample(X_trainCV, Y_trainCV)
        elif sampling_method == "smote":
            X_train_resampled, Y_train_resampled = SMOTE().fit_sample(X_trainCV, Y_trainCV)
        elif sampling_method == "random":
            X_train_resampled, Y_train_resampled = RandomOverSampler(random_state=0).fit_sample(X_trainCV, Y_trainCV)
        elif sampling_method == "smoteenn":
            X_train_resampled, Y_train_resampled = SMOTEENN().fit_sample(X_trainCV, Y_trainCV)
        else:
            X_train_resampled, Y_train_resampled = X_trainCV, Y_trainCV
        if verbose:
            print("Oversampling completed")
            print("Time Taken: {:.2f}".format(time.time()-curr_time))
            print("Size of Oversampled data: {}".format(X_train_resampled.shape))
        curr_time = time.time() 
        lgb_train = lgb.Dataset(data=X_train_resampled, label=Y_train_resampled, feature_name=feature_labels)
        gbm = lgb.train(params, lgb_train, num_boost_round=30)
        print("Time taken: {:.2f}".format(time.time()-curr_time))
        Y_probs = gbm.predict(X_testCV)
        result = evaluate(Y_testCV, Y_probs)
        print(result)
        pickle.dump(result, open(os.path.join(results_dir, "lgbm_{}_fold_{}.p".format(sampling_method, k+1)), "wb" )) 

    print("Training took {:.2f}s.".format(time.time()-start_time))
    print("Finished.") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb New User Booking Classification")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-r', help='results save directory', dest='results_dir', type=str, default="results")
    parser.add_argument('-s', help='sampling method', dest='sampling_method', type=str, default="smote", 
                        choices=["random", "smote", "smoteenn", "adasyn", "none"])
    parser.add_argument('-k', help='number of CV folds', dest='k_folds', type=int, default=5)
    parser.add_argument('--device', help='device', dest='device', type=str, default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    train(args.sampling_method, args.k_folds, args.data_dir, args.results_dir, args.device, verbose=True)
    
