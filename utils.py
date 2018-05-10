import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    metrics = ["ndcg", "acc", "precision", "recall", "F1"]
    y_pred = np.argmax(y_score, axis=1)
    ndcg = ndcg_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro') 
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    vals = [ndcg, acc, precision, recall, f1]
    return {metric:val for metric, val in zip(metrics, vals)}

