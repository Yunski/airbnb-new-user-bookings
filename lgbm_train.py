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

from utils import get_train, evaluate, save_examples

def train(sampling_method, k_folds, data_dir, results_dir, device='cpu', use_international=False, verbose=True):
    model = 'lgbm'
    start_time = time.time()
    if verbose:
        print("Using device: {}".format(device))
        print("Reading train data in...")
        if use_international:
            print("Using international class.")
    X_train, Y_train, feature_labels = get_train(data_dir, one_hot=False, use_international=use_international)
    categorical_feature = ['age_bucket', 'gender', 'signup_method', 'signup_flow', 'language', 
                           'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                           'signup_app', 'first_device_type', 'first_browser']
    if verbose:
        print("Successfully loaded data")

    print("Starting Cross-Validation with {} folds...".format(k_folds))
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(X_train)
    params = {
        'task': 'train',
        'objective': 'multiclass',
        'num_class': 12,
        'num_leaves': 31,
        'lambda_l2': 0.1,
        'learning_rate': 0.3,
        'feature_fraction': 0.9,
        'min_child_weight': 1.0,
        'device': device,
        'gpu_device_id': 0,
        'gpu_platform_id': 0,
        'max_bin': 63,
        'verbose': 0
    }
    if use_international:
        params['objective'] = 'binary'
        del params["num_class"]
     
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
            X_train_resampled, Y_train_resampled = RandomOverSampler().fit_sample(X_trainCV, Y_trainCV)
        elif sampling_method == "smoteenn":
            X_train_resampled, Y_train_resampled = SMOTEENN().fit_sample(X_trainCV, Y_trainCV)
        else:
            X_train_resampled, Y_train_resampled = X_trainCV, Y_trainCV

        if verbose:
            print("Oversampling completed")
            print("Time Taken: {:.2f}".format(time.time()-curr_time))
            print("Size of Oversampled data: {}".format(X_train_resampled.shape))
            print("{} model(s) selected for classification".format(model))

        curr_time = time.time()
        lgb_train = lgb.Dataset(data=X_train_resampled, label=Y_train_resampled, 
                                feature_name=feature_labels, categorical_feature=categorical_feature)
        clf = lgb.train(params, lgb_train, num_boost_round=30) 
        print("Time taken: {:.2f}".format(time.time()-curr_time))
        Y_probs = clf.predict(X_testCV) 
        result = evaluate(Y_testCV, Y_probs)
        print(result)
        feature_imp = clf.feature_importance(importance_type='gain') 
        feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
        pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "wb" ))
        pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
        save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)

    print("Training took {:.2f}s.".format(time.time()-start_time))
    print("Finished.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightGBM Training Script")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-r', help='results save directory', dest='results_dir', type=str, default="results")
    parser.add_argument('-s', help='sampling method', dest='sampling_method', type=str, default="smote", 
                        choices=["random", "smote", "adasyn", "smoteenn", "none"])
    parser.add_argument('-k', help='number of CV folds', dest='k_folds', type=int, default=5)
    parser.add_argument('--international', help='group minority classes into international class', dest='international', action='store_true')
    parser.add_argument('--device', help='device', dest='device', type=str, default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if args.international:
        results_dir = os.path.join(args.results_dir, "binary")
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
    else:
        results_dir = os.path.join(args.results_dir, "multiclass")
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
    train(args.sampling_method, args.k_folds, args.data_dir, results_dir, args.device, args.international, verbose=True)

