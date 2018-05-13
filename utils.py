import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

def get_train(data_dir, one_hot=True, use_international=False):
    if one_hot:
        train = pd.read_feather(os.path.join(data_dir, "train.feather"))
    else:
        train = pd.read_feather(os.path.join(data_dir, "train_lgb.feather"))
    train_features = train.as_matrix()
    feature_labels = train.columns.copy().tolist()
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    if use_international:
        train_labels[train_labels > 2] = 2 
    train_features, train_labels = shuffle(train_features, train_labels)
    return train_features, train_labels, feature_labels

def get_test(data_dir, one_hot=True):
    if one_hot:
        test_features = pd.read_feather(os.path.join(data_dir, "test.feather")).as_matrix()
    else:
        test_features = pd.read_feather(os.path.join(data_dir, "test_lgb.feather")).as_matrix()
    return test_features

def get_ids(data_dir):
    train_ids = pd.read_csv(os.path.join(data_dir, "train_ids.csv"))['id'].tolist()
    test_ids = pd.read_csv(os.path.join(data_dir, "test_ids.csv"))['id'].tolist()
    return train_ids, test_ids

def get_country_names(data_dir):
    return np.load(os.path.join(data_dir, "country_labels.npy"))

def ndcg_score(y_true, y_score, k=5): 
    order = np.argsort(-y_score[:,:k], axis=1)
    rel = np.array([(pred == truth).astype(int) for pred, truth in zip(order, y_true)])
    discounts = np.log2(np.arange(rel.shape[1]) + 2)
    return np.sum((2**rel-1) / discounts, axis=1)   

def evaluate(y_true, y_score):
    metrics = ["ndcg", "ndcg_fold", "acc", "confusion_matrix", 
               "precision", "recall", "num_class_p", "f1", 
               "macro_precision", "macro_recall", "macro_f1"]
    y_pred = np.argmax(y_score, axis=1)
    ndcg = ndcg_score(y_true, y_score)
    ndcg_fold = np.mean(ndcg)
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tp = np.diag(conf_matrix).astype(np.float64)
    col_sum = np.sum(conf_matrix, axis=0)
    row_sum = np.sum(conf_matrix, axis=1)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum!=0)
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum!=0)
    num_classes_in_precision = np.sum(col_sum > 0)
    f1_num = 2 * (precision * recall)
    f1_denom = precision + recall
    f1 = np.divide(f1_num, f1_denom, out=np.zeros_like(f1_num), where=f1_denom!=0)
    macro_precision = np.mean(precision) 
    macro_recall = np.mean(recall)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    vals = [ndcg, ndcg_fold, acc, conf_matrix, precision, recall, 
            num_classes_in_precision, f1, macro_precision, macro_recall, macro_f1]
    return {metric: np.around(val, 4) for metric, val in zip(metrics, vals)}

def save_examples(X, y_true, y_score, model, sampling_method, fold, save_dir="results"):
    label_counts = np.bincount(y_true)
    y_pred = np.argmax(y_score, axis=1) 
    correct_pred = np.array(y_true == y_pred)
    correct_samples = X[correct_pred]
    classes_with_correct_pred = y_true[correct_pred]
    counts_correct = np.bincount(classes_with_correct_pred).astype(np.float64)
    top_three_classes = np.argsort(-counts_correct / label_counts[:len(counts_correct)])[:3]
    samples_to_save = []
    for label in top_three_classes:
        samples = correct_samples[classes_with_correct_pred == label]
        sample = samples[np.random.choice(np.arange(len(samples)))]
        samples_to_save.append({'features': sample, 'label': label})
    pickle.dump(samples_to_save, open(os.path.join(save_dir, "{}_{}_fold_{}_correct_examples.p".format(model, sampling_method, fold)), "wb"))
    incorrect_pred = np.array(y_true != y_pred)
    incorrect_samples = X[incorrect_pred]
    classes_with_incorrect_pred = y_true[incorrect_pred]
    counts_incorrect = np.bincount(classes_with_incorrect_pred).astype(np.float64)
    top_three_classes = np.argsort(-counts_incorrect / label_counts[:len(counts_incorrect)])[:3]
    samples_to_save = []
    for label in top_three_classes:
        samples = incorrect_samples[classes_with_incorrect_pred == label]
        sample = samples[np.random.choice(np.arange(len(samples)))]
        samples_to_save.append({'features': sample, 'label': label})
    pickle.dump(samples_to_save, open(os.path.join(save_dir, "{}_{}_fold_{}_incorrect_examples.p".format(model, sampling_method, fold)), "wb"))
    

