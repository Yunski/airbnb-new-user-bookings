import argparse
import csv
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb 

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from utils import get_train, get_test, get_ids, get_country_names

def submission(model, sampling_method, data_dir, results_dir, device='cpu', verbose=True):
    if verbose:
        print("Using device: {}".format(device))
        print("Reading train data in...")
    if model == 'lgbm':
        X_train, Y_train, feature_labels = get_train(data_dir, one_hot=False)
    else:
        X_train, Y_train, feature_labels = get_train(data_dir)

    X_test = get_test(data_dir)
    train_ids, test_ids = get_ids(data_dir)
    country_names = get_country_names(data_dir)

    if verbose:
        print("Successfully loaded data")

    lgbm_params = {
        'task': 'train',
        'objective': 'multiclass',
        'num_class': 12,
        'num_leaves': 31,
        'learning_rate': 0.3,
        'lambda_l2': 1.0,
        'feature_fraction': 0.9,
        'min_child_weight': 1.0,
        'device': device,
        'gpu_device_id': 0,
        'gpu_platform_id': 0,
        'max_bin': 63,
        'verbose': 0
    }

    if device == 'cpu':
        xgb_params = {"objective": "multi:softprob", 
                  "num_class": 12, 
                  "tree_method": "hist", 
                  "colsample_bytree": 0.9,
                  "n_jobs": 2,
                  "silent": 1}
    else:
        xgb_params = {"objective": "multi:softprob", 
                  "num_class": 12,
                  "tree_method": "gpu_hist",
                  "colsample_bytree": 0.9,
                  "gpu_id": 0,
                  "max_bin": 16,
                  "silent": 1}
    if verbose:
        print("{} sampling process started...".format(sampling_method))
    curr_time = time.time()

    if sampling_method == "adasyn":
        X_train_resampled, Y_train_resampled = ADASYN().fit_sample(X_train, Y_train)
    elif sampling_method == "smote":
        X_train_resampled, Y_train_resampled = SMOTE().fit_sample(X_train, Y_train)
    elif sampling_method == "random":
        X_train_resampled, Y_train_resampled = RandomOverSampler().fit_sample(X_train, Y_train)
    elif sampling_method == "smoteenn":
        X_train_resampled, Y_train_resampled = SMOTEENN().fit_sample(X_train, Y_train)
    else:
        X_train_resampled, Y_train_resampled = X_train, Y_train

    if verbose:
        print("Oversampling completed")
        print("Time Taken: {:.2f}".format(time.time()-curr_time))
        print("Size of Oversampled data: {}".format(X_train_resampled.shape))
        print("{} selected for classification".format(model))


    curr_time = time.time()
    if model == 'lgbm':
        categorical_feature = ['age_bucket', 'gender', 'signup_method', 'signup_flow', 'language',
                               'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                               'signup_app', 'first_device_type', 'first_browser']
        lgb_train = lgb.Dataset(data=X_train_resampled, label=Y_train_resampled, feature_name=feature_labels, categorical_feature=categorical_feature)
        clf = lgb.train(lgbm_params, lgb_train, num_boost_round=30)
        print("Time taken: {:.2f}".format(time.time()-curr_time)) 
        Y_probs = clf.predict(X_test)
        order = np.argsort(-Y_probs[:,:5], axis=1)
    else:
        X_train_xgb = xgb.DMatrix(X_train_resampled, Y_train_resampled, feature_names=feature_labels)
        X_test_xgb  = xgb.DMatrix(X_test, feature_names=feature_labels)
        clf = xgb.train(xgb_params, X_train_xgb, 30)
        print("Time taken: {:.2f}".format(time.time()-curr_time))
        Y_probs = clf.predict(X_test_xgb)
        order = np.argsort(-Y_probs[:,:5], axis=1)
    
    print("Generating submission csv...")
    with open(os.path.join(results_dir, 'submission_{}.csv'.format(model)), 'w') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id','country'])
        for i in range(len(test_ids)):
            for k in range(5):
                writer.writerow([test_ids[i], country_names[order[i, k]]])
    print("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kaggle Submission Script")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-r', help='results save directory', dest='results_dir', type=str, default="results")
    parser.add_argument('-m', help='model', dest="model", type=str, default="lgbm", choices=["lgbm", "xgb"])
    parser.add_argument('-s', help='sampling method', dest='sampling_method', type=str, default="smote", 
                        choices=["random", "smote", "adasyn", "smoteenn", "none"])
    parser.add_argument('--device', help='device', dest='device', type=str, default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    submission(args.model, args.sampling_method, args.data_dir, args.results_dir, args.device, verbose=True)

