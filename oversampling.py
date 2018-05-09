import numpy as np
import pandas as pd
import argparse
import time
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.ensemble import EasyEnsemble, BalancedBaggingClassifier, BalanceCascade
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


## Over-Sampling
# Random Over Sampling
def preprocess(data_dir, model, oversampling_method, k_folds, verbose=True):
    if verbose:
        print("Reading train data in...")
    X_train = pd.read_feather(os.path.join(data_dir, "train.feather")).as_matrix()
    Y_train = np.load(os.path.join(data_dir, "train_labels.npy"))
    if verbose:
        print("Successfully loaded data")

    print("Starting Cross-Validation with {} folds...".format(k_folds))
    counter = 1
    curr_time = time.time()
    avg_df = 0
    avg_rr = 0
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(X_train)
    for train_index, test_index in (kf.split(X_train)):
        if verbose:
            print("Processing Fold {} out of {}".format(counter, k_folds))
            print(" ")
            counter += 1
            
        X_trainCV, X_testCV = X_train[train_index], X_train[test_index]
        Y_trainCV, Y_testCV = Y_train[train_index], Y_train[test_index]

        if verbose:
            print("{} Oversampling Process started...".format(oversampling_method))
        curr_time = time.time()
        if (oversampling_method == "ADASYN"):
            X_train_resampled, Y_train_resampled = ADASYN().fit_sample(X_trainCV, Y_trainCV)
        elif (oversampling_method == "SMOTE"):
            X_train_resampled, Y_train_resampled = SMOTE().fit_sample(X_trainCV, Y_trainCV)
        elif (oversampling_method == "none"):
            X_train_resampled, Y_train_resampled = X_train, Y_train
        else:
            X_train_resampled, Y_train_resampled = RandomOverSampler(random_state=0).fit_sample(X_trainCV, Y_trainCV)

        if verbose:
            print("Oversampling completed")
            print("Time Taken for Oversampling is {}".format(time.time()-curr_time))
            print("Size of Oversampled data: {}".format(X_train_resampled.shape))
            print(" ")

        if verbose:
            print("{} model(s) selected for classification".format(model))

        curr_time = time.time()
        if (model == "decisiontree"):
            clf = DecisionTreeClassifier().fit(X_train_resampled, Y_train_resampled)
            y_pred = clf.predict(X_testCV)
            accuracy = accuracy_score(Y_testCV, y_pred, normalize=True)
            avg_df += accuracy
            if verbose:
                print("Accuracy for DecisionTreeClassifier is {}".format(accuracy))
                print("Time taken: {}".format(time.time()-curr_time))

        elif (model == "ridge"):
            clf = LogisticRegression(penalty="l2", C=1).fit(X_train_resampled, Y_train_resampled)
            y_pred = clf.predict(X_testCV)
            accuracy = accuracy_score(Y_testCV, y_pred, normalize=True)
            avg_rr += accuracy
            if verbose:
                print("Accuracy for RidgeClassifier is {}".format(accuracy))
                print("Time taken: {}".format(time.time()-curr_time))
        else:
            print("DecisionTreeClassifier Running...")
            clf_dt = DecisionTreeClassifier().fit(X_train_resampled, Y_train_resampled)
            y_pred = clf_dt.predict(X_testCV)
            accuracy = accuracy_score(Y_testCV, y_pred, normalize=True)
            avg_df += accuracy
            if verbose:
                print("Accuracy for DecisionTreeClassifier is {}".format(accuracy))
                print("Time taken: {}".format(time.time()-curr_time))

            print("RidgeClassifier...")
            curr_time = time.time()
            clf_rr = LogisticRegression(penalty="l2", C=1).fit(X_train_resampled, Y_train_resampled)
            y_pred = clf_rr.predict(X_testCV)
            accuracy = accuracy_score(Y_testCV, y_pred, normalize=True)
            if verbose:
                print("Accuracy for RidgeClassifier is {}".format(accuracy))
                print("Time taken: {}".format(time.time()-curr_time))
            avg_rr += accuracy
            end_time = start - time.time()

    print("Total time taken for xxx is {}".format(end_time))
    if (model == "both"):
        print("DecisionTreeClassifier CV accuracy: {}".format(avg_df/5.0))
        print("RidgeClassifier CV accuracy: {}".format(avg_rr/5.0))
    else:
        if (model == "ridge"):
            avg_df = avg_rr
        print("{} CV accuracy: {}".format(model, avg_df))
    os.system('say "your program has finished"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb Data Preprocessing")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-m', help='classification model', dest = "model", type=str, default="both", choices=["both", "decisiontree", "ridge"])
    parser.add_argument('-o', help='oversampling method', dest='oversampling_method', type=str, default="Random", choices=["random", "SMOTE", "ADASYN", "none"])
    parser.add_argument('-k', help='number of CV folds', dest='K_folds', type=int, default=5)
    args = parser.parse_args()
    preprocess(args.data_dir,args.model, args.oversampling_method, args.K_folds, verbose=True)
