# Library-like script providing pool of all aggregating STS methods
# Focused on sklearn models

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import re
from os import listdir, getcwd
from os.path import isfile, join
import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from dataset_modification_scripts.dataset_wrapper import input_folder

# Prepare regexes to be used in this script
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

# Load list of existing dataset files
input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]

# Prepare list of regression sklearn models to be added to pool
model_types = [
    {
        "name": "linear_regression",
        "model": LinearRegression,
        "args": {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            #'positive': [True, False]
        }
    },
    {
        "name": "support_vector_regression",
        "model": SVR,
        "args": {
            # Most significant ones
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'] + [y/(10 ** x) for x in range(1, 4) for y in [1, 2, 5]],   # 0.1, 0.2, 0.5, 0.01, 0.02, 0.05 ...
            'C': [y/(10 ** x) for x in [0, 1] for y in [1, 2, 3, 5, 7, 9]] + [10.0],              # 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 10.0

            # Less significant ones
            'degree': list(map(lambda x: x * 2 + 1, range(1, 10))),                               # 3, 5, 7, 9 ... 21,
            'epsilon': [y/(10 ** x) for x in range(1, 3) for y in [1, 2, 3]] + [0.05, 0.07, 0.09],# 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09
            'shrinking': [True, False],
            #'max_iter': [-1] + [x * 50 for x in range(1, 9)],                                     # 50, 100, 150, 200, 250, 300, 350, 400
            'coef0': [0] + [y/(10 ** x) for x in range(1, 4) for y in [1, 2, 3]]                  # 0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.001, 0.002, 0.003
        }
    },
    {
        "name": "decision_tree_regression",
        "model": DecisionTreeRegressor,
        "args": {
            'splitter': ['random', 'best'],
            'max_depth': list(range(5, 20)) + [None],
            'min_samples_split': list(range(2, 50)),
            'min_samples_leaf': list(range(1, 30)),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_leaf_nodes': list(range(2, 10))
        }
    }
]
