import argparse
import os
import pickle
import numpy as np

def summary(model, sampling_method, k_folds, results_dir, verbose=True):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airbnb New User Booking Train Results Summarizer")
    parser.add_argument('model', help='model', choices=["logistic", "tree", "forest", "ada", "xgb", "lgbm"])
    parser.add_argument('sampling_method', help='sampling method', choices=["random", "smote", "adasyn", "smoteenn", "none"])
    parser.add_argument('-k', help='number of CV folds', dest='k_folds', type=int, default=5)
    parser.add_argument('-r', help='results save directory', dest='results_dir', type=str, default="results")
    args = parser.parse_args()

    summary(args.model, args.sampling_method, args.k_folds, args.results_dir, verbose=True)
 
