import argparse
import os
import pickle
import numpy as np

from utils import get_train, get_country_names

def summary(model, sampling_method, k_folds, use_international, data_dir, results_dir, verbose=True):
    print("model: {} - sampling method: {}".format(model, sampling_method))
    aggregate = {}
    for k in range(k_folds):
        result = pickle.load(open(os.path.join(results_dir, "{}_{}_fold_{}.p".format(model, sampling_method, k+1)), "rb"))
        print("Fold {}".format(k+1))
        for key, val in result.items():
            print(key)
            print(val)
            if key not in aggregate:
                if type(val) is np.float32 or type(val) is np.float64 or type(val) is float:
                    aggregate[key] = val
            else:
                if type(val) is np.float32 or type(val) is np.float64 or type(val) is float:
                    aggregate[key] += val
    aggregate = {key: val / k_folds for key, val in aggregate.items()}
    print("Aggregate")
    print(aggregate)
    if model == 'logistic':
        print("feature importance not implemented for logistic regression")
        return
    features, labels, feature_labels = get_train(data_dir, use_international=use_international)
    country_names = get_country_names(data_dir)
    if use_international:
        country_names = country_names[:2].tolist() + ['international']
    for k in range(k_folds):
        print("Fold {}".format(k+1))
        correct_examples = pickle.load(open(os.path.join(results_dir, "{}_{}_fold_{}_correct_examples.p".format(model, sampling_method, k+1)), "rb"))
        incorrect_examples = pickle.load(open(os.path.join(results_dir, "{}_{}_fold_{}_incorrect_examples.p".format(model, sampling_method, k+1)), "rb"))
        feature_imp = pickle.load(open(os.path.join(results_dir, "{}_{}_feature_imp_fold_{}.p".format(model, sampling_method, k+1)), "rb"))
        top_20 = [(label, feature_imp[label]) for label in sorted(feature_imp, key=feature_imp.get, reverse=True)][:20]
        print("correct examples\n")
        for example in correct_examples:
            print("{} features\n".format(country_names[example['label']]))
            feature_dict = { label: feature for label, feature in zip(feature_labels, example['features']) }
            for label, weight in top_20:
                print("{},{}".format(label, feature_dict[label]))
            print("")
        print("\nincorrect examples\n")
        for example in incorrect_examples:
            print("{} features\n".format(country_names[example['label']])) 
            feature_dict = { label: feature for label, feature in zip(feature_labels, example['features']) }
            for label, weight in top_20:
                print("{},{}".format(label, feature_dict[label]))
            print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb New User Booking Train Results Summarizer")
    parser.add_argument('model', help='model', choices=["logistic", "tree", "forest", "ada", "xgb", "lgbm"])
    parser.add_argument('sampling_method', help='sampling method', choices=["random", "smote", "adasyn", "smoteenn", "none"])
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    parser.add_argument('-r', help='results save directory', dest='results_dir', type=str, default="results")
    parser.add_argument('-k', help='number of CV folds', dest='k_folds', type=int, default=5)
    parser.add_argument('--international', help='group minority classes into international class', dest='international', action='store_true')
    args = parser.parse_args()
    if args.international:
        results_dir = os.path.join(args.results_dir, "multiclass-3")
    else:
        results_dir = os.path.join(args.results_dir, "multiclass-12")
    summary(args.model, args.sampling_method, args.k_folds, args.international, args.data_dir, results_dir, verbose=True)
 
