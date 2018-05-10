import numpy as np
import pandas as pd
import argparse
import pickle
import time
import os
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from utils import get_train, evaluate

def train(model, oversampling_method, k_folds, data_dir, results_dir, device='cpu', verbose=True):
    start_time = time.time()
    if verbose:
        print("Using device: {}".format(device))
        print("Reading train data in...")
    X_train, Y_train, _ = get_train(data_dir)
    if verbose:
        print("Successfully loaded data")

    print("Starting Cross-Validation with {} folds...".format(k_folds))
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(X_train)
    if device == 'cpu':
        params = {"objective": "multi:softmax", "num_class": 12, "tree_method": "hist", "silent": 1}
    else:
        params = {"objective": "multi:softmax", 
                  "num_class": 12,
                  "gpu_id": 0,
                  "max_bin": 16,
                  "tree_method": "gpu_hist",
                  "silent": 1}
    if verbose:
        params["silent"] = 0

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
            print("{} model(s) selected for classification".format(model))

        curr_time = time.time()
        if model == "tree":
            clf = DecisionTreeClassifier().fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)
            pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" )) 
        elif model == "logistic":
            clf = LogisticRegression(penalty="l2", C=1).fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)
            pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" )) 
        elif model == "xgb":
            X_train_xgb = xgb.DMatrix(X_train_resampled, Y_train_resampled)
            X_test_xgb  = xgb.DMatrix(X_testCV)
            clf = xgb.train(params, X_train_xgb, 20)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)
            pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
        elif model == "randomforest":
            clf = RandomForestClassifier(n_estimators=22).fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)
            pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
        elif model == "svm":
            clf = LinearSVC().fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_dist = -np.abs(clf.decision_function(X_testCV))
            result = evaluate(Y_testCV, Y_dist) 
            pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
        else:
            models = ["xgb", "svm", "randomforest", "tree", "logistic"]
            for model in models:
                print("Training {}...".format(model))
                curr_time = time.time()
                if model == "xgb":
                    X_train_xgb = xgb.DMatrix(X_train_resampled, Y_train_resampled)
                    X_test_xgb  = xgb.DMatrix(X_testCV)      
                    clf = xgb.train(params, X_train_xgb, 20)
                    print("Time taken for {}: {:.2f}".format(model, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV) 
                    result = evaluate(Y_testCV, Y_probs)
                    pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
                elif model == "svm":
                    clf = LinearSVC().fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(model, time.time()-curr_time))
                    Y_dist = -np.abs(clf.decision_function(X_testCV)) 
                    result = evaluate(Y_testCV, Y_dist)
                    pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
                elif model == "randomforest":
                    clf = RandomForestClassifier(n_estimators=22).fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(model, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)
                    pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
                elif model == "tree":
                    clf = DecisionTreeClassifier().fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(model, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)
                    pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
                else:
                    clf = LogisticRegression(penalty="l2", C=1).fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(model, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)
                    pickle.dump(result, open(os.path.join(results_dir, "{}_fold_{}.p".format(model, k+1)), "wb" ))
    print("Training took {:.2f}s.".format(time.time()-start_time))
    print("Finished.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb New User Booking Classification")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-s', help='results save directory', dest='results_dir', type=str, default="data")
    parser.add_argument('-m', help='model', dest="model", type=str, default="all", 
                        choices=["all", "logistic", "svm", "tree", "randomforest", "xgb"])
    parser.add_argument('-o', help='oversampling method', dest='oversampling_method', type=str, default="smote", 
                        choices=["random", "smote", "adasyn", "none"])
    parser.add_argument('-k', help='number of CV folds', dest='k_folds', type=int, default=5)
    parser.add_argument('--device', help='device', dest='device', type=str, default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    train(args.model, args.oversampling_method, args.k_folds, args.data_dir, args.results_dir, args.device, verbose=True)

