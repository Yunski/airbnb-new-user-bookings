import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

def get_train(data_dir, one_hot=True):
    if one_hot:
        train = pd.read_feather(os.path.join(data_dir, "train.feather"))
    else:
        train = pd.read_feather(os.path.join(data_dir, "train_lgb.feather"))
    train_features = train.as_matrix()
    feature_labels = train.columns.copy().tolist()
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
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
    metrics = ["ndcg", "ndcg_fold", "acc", "precision", "recall", "f1", "macro_precision", "macro_recall", "macro_f1"]
    y_pred = np.argmax(y_score, axis=1)
    ndcg = ndcg_score(y_true, y_score)
    ndcg_fold = np.mean(ndcg)
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tp = np.diag(conf_matrix).astype(np.float32)
    col_sum = np.sum(conf_matrix, axis=0)
    row_sum = np.sum(conf_matrix, axis=1)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum!=0)
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum!=0)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    f1_num = 2 * (precision * recall)
    f1_denom = precision + recall
    f1 = np.divide(f1_num, f1_denom, out=np.zeros_like(f1_num), where=f1_denom!=0)
    macro_f1 = np.mean(f1)
    vals = [ndcg, ndcg_fold, acc, precision, recall, f1, macro_precision, macro_recall, macro_f1]
    return {metric: np.around(val, 4) for metric, val in zip(metrics, vals)}

