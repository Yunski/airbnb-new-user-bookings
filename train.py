import numpy as np
import pandas as pd
import argparse
import pickle
import time
import os
import lightgbm as lgb 
import xgboost as xgb

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from utils import get_train, evaluate, save_examples

def train(model, sampling_method, k_folds, data_dir, results_dir, device='cpu', use_international=False, verbose=True):
    start_time = time.time()
    if verbose:
        print("Using device: {}".format(device))
        print("Reading train data in...")
        if use_international:
            print("Using international class.")

    X_train, Y_train, feature_labels = get_train(data_dir, use_international=use_international)
 
    if verbose:
        print("Successfully loaded data")

    print("Starting Cross-Validation with {} folds...".format(k_folds))
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(X_train)
    lgbm_params = {
        'task': 'train',
        'objective': 'multiclass',
        'num_class': 12,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'lambda_l2': 1.0,
        'feature_fraction': 0.9,
        'min_child_weight': 1.0,
        'device': device,
        'gpu_device_id': 0,
        'gpu_platform_id': 0,
        'max_bin': 63,
        'verbose': 0
    }
    if use_international:
        lgbm_params['objective'] = 'binary'
        del lgbm_params["num_class"]
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
    if use_international:
        xgb_params["objective"] = "binary:logistic"
        del xgb_params["num_class"]

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
        if model == "tree":
            clf = DecisionTreeClassifier().fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)
            print(result)
            feature_imp = clf.feature_importances_
            feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
            pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "wb" ))
            pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
            save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)
        elif model == "logistic":
            clf = LogisticRegression(penalty="l2", C=1).fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            assert(np.all(np.argmax(Y_probs, axis=1) == clf.predict(X_testCV)))
            result = evaluate(Y_testCV, Y_probs)
            print(result)
            pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
            save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)
        elif model == "xgb":
            X_train_xgb = xgb.DMatrix(X_train_resampled, Y_train_resampled, feature_names=feature_labels)
            X_test_xgb  = xgb.DMatrix(X_testCV, feature_names=feature_labels)
            clf = xgb.train(xgb_params, X_train_xgb, 30)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict(X_test_xgb)
            result = evaluate(Y_testCV, Y_probs)
            print(result)
            feature_imp = clf.get_score(importance_type='gain')
            pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "wb" ))
            pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
            save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)
        elif model == "lgbm":
            lgb_train = lgb.Dataset(data=X_train_resampled, label=Y_train_resampled, feature_name=feature_labels)
            clf = lgb.train(lgbm_params, lgb_train, num_boost_round=30) 
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict(X_testCV) 
            result = evaluate(Y_testCV, Y_probs)
            print(result)
            feature_imp = clf.feature_importance(importance_type='gain') 
            feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
            pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "wb" ))
            pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
            save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)
        elif model == "ada":
            clf = AdaBoostClassifier(n_estimators=30).fit(X_train_resampled, Y_train_resampled)
            print("Time taken for {}: {:.2f}".format(model, time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)  
            print(result)
            feature_imp = clf.feature_importances_
            feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
            pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "wb" ))
            pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
            save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)
        elif model == "forest":
            clf = RandomForestClassifier(n_estimators=30, n_jobs=2).fit(X_train_resampled, Y_train_resampled)
            print("Time taken: {:.2f}".format(time.time()-curr_time))
            Y_probs = clf.predict_proba(X_testCV)
            result = evaluate(Y_testCV, Y_probs)
            print(result)
            feature_imp = clf.feature_importances_
            feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
            pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "wb" ))
            pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "wb" )) 
            save_examples(X_testCV, Y_testCV, Y_probs, model, sampling_method, k+1, save_dir=results_dir)
        else:
            models = ["lgbm", "xgb", "ada", "forest", "tree", "logistic"] # for category codes instead of one hot, use lgbm_train.py
            for m in models:
                print("Training {}...".format(m))
                curr_time = time.time()
                if m == "xgb":
                    X_train_xgb = xgb.DMatrix(X_train_resampled, Y_train_resampled, feature_names=feature_labels)
                    X_test_xgb  = xgb.DMatrix(X_testCV, feature_names=feature_labels)      
                    clf = xgb.train(xgb_params, X_train_xgb, 30)
                    print("Time taken for {}: {:.2f}".format(m, time.time()-curr_time))
                    Y_probs = clf.predict(X_test_xgb) 
                    result = evaluate(Y_testCV, Y_probs)
                    print(result)
                    feature_imp = clf.get_score(importance_type='gain')
                    pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(m, sampling_method, k+1)), "wb" ))
                    pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(m, sampling_method, k+1)), "wb" )) 
                    save_examples(X_testCV, Y_testCV, Y_probs, m, sampling_method, k+1, save_dir=results_dir)
                elif m == "lgbm":
                    lgb_train = lgb.Dataset(data=X_train_resampled, label=Y_train_resampled, feature_name=feature_labels)
                    clf = lgb.train(lgbm_params, lgb_train, num_boost_round=30) 
                    print("Time taken for {}: {:.2f}".format(m, time.time()-curr_time))
                    Y_probs = clf.predict(X_testCV) 
                    result = evaluate(Y_testCV, Y_probs)
                    print(result)
                    feature_imp = clf.feature_importance(importance_type='gain') 
                    feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
                    pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(m, sampling_method, k+1)), "wb" ))
                    pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(m, sampling_method, k+1)), "wb" )) 
                    save_examples(X_testCV, Y_testCV, Y_probs, m, sampling_method, k+1, save_dir=results_dir)
                elif m == "ada":
                    clf = AdaBoostClassifier(n_estimators=30).fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(m, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)  
                    print(result)
                    feature_imp = clf.feature_importances_
                    feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
                    pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(m, sampling_method, k+1)), "wb" ))
                    pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(m, sampling_method, k+1)), "wb" )) 
                    save_examples(X_testCV, Y_testCV, Y_probs, m, sampling_method, k+1, save_dir=results_dir)
                elif m == "forest":
                    clf = RandomForestClassifier(n_estimators=30, n_jobs=2).fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(m, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)
                    print(result)
                    feature_imp = clf.feature_importances_
                    feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
                    pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(m, sampling_method, k+1)), "wb" ))
                    pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(m, sampling_method, k+1)), "wb" )) 
                    save_examples(X_testCV, Y_testCV, Y_probs, m, sampling_method, k+1, save_dir=results_dir)
                elif m == "tree":
                    clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=5).fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(m, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)
                    print(result)
                    feature_imp = clf.feature_importances_
                    feature_imp = {label: imp for label, imp in zip(feature_labels, feature_imp)}
                    pickle.dump(feature_imp, open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(m, sampling_method, k+1)), "wb" ))
                    pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(m, sampling_method, k+1)), "wb" )) 
                    save_examples(X_testCV, Y_testCV, Y_probs, m, sampling_method, k+1, save_dir=results_dir)
                else:
                    clf = LogisticRegression(penalty="l2").fit(X_train_resampled, Y_train_resampled)
                    print("Time taken for {}: {:.2f}".format(m, time.time()-curr_time))
                    Y_probs = clf.predict_proba(X_testCV)
                    result = evaluate(Y_testCV, Y_probs)
                    print(result)
                    pickle.dump(result, open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(m, sampling_method, k+1)), "wb" )) 
                    save_examples(X_testCV, Y_testCV, Y_probs, m, sampling_method, k+1, save_dir=results_dir)

    print("Training took {:.2f}s.".format(time.time()-start_time))
    print("Finished.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb New User Booking Classification")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-r', help='results save directory', dest='results_dir', type=str, default="results")
    parser.add_argument('-m', help='model', dest="model", type=str, default="all", 
                        choices=["all", "logistic", "tree", "forest", "ada", "xgb", "lgbm"])
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
    train(args.model, args.sampling_method, args.k_folds, args.data_dir, results_dir, args.device, args.international, verbose=True)

