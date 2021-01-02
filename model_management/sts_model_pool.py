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

from model_management.sts_method_value_persistor import input_folder
from evaluation.evaluate_regression_metrics import find_best_methods, group_by_method, get_dataset_metrics
from complex_similarity_methods.regression_methods_core import train_n_test
from model_management.sts_method_wrappers import STSModel

# Prepare regexes to be used in this script
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

# Load list of existing dataset files
input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]

# Prepare list of regression sklearn models to be added to pool
model_types = [
    {
        "name": "linear_regression",
        "model": LinearRegression,
        "args": {}
    },
    {
        "name": "support_vector_regression",
        "model": SVR,
        "args": {}
    },
    {
        "name": "decision_tree_regression",
        "model": DecisionTreeRegressor,
        "args": {}
    }
]

# Initialize the pool of models
sts_model_pool = {}
# Loop over each dataset and prepare models for each dataset. They are won't be trained nor loaded yet.
for dataset in input_dataset_files:
    # Create entry in model pool for current dataset
    sts_model_pool[dataset] = {}
    # Get metrics of simple STS values for given dataset
    evaluation_values = get_dataset_metrics(dataset)
    # For each method, determine best param configuration
    input_names = find_best_methods(group_by_method(evaluation_values))

    # Loop over available model types and prepare them in wrapper for this dataset
    for model_type in model_types:
        # Create the wrapper instance
        sts_model = STSModel(
            model_type["name"],
            model_type["model"](),
            model_type["args"],
            input_names,
            train_n_test
        )
        # Add the wrapper instance to pool
        sts_model_pool[dataset][sts_model.name] = sts_model
