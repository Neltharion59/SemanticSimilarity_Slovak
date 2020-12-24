from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from complex_similarity_methods.regression_methods_core import train_n_test
from model_management.sts_method_wrappers import STSModel
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


#def add_to_model_pool(method_name, method_arg_possibilites, method_function, model_pool):
#    for arg_variation in list(all_arg_variations(method_arg_possibilites, {})):
#        sts_method = STSModel(method_name, method_function, arg_variation)
#        model_pool[sts_method.name] = sts_method


dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]


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

sts_model_pool = {}
for dataset in input_dataset_files:
    sts_model_pool[dataset] = {}
    evaluation_values = get_dataset_metrics(dataset)
    input_names = find_best_methods(group_by_method(evaluation_values))

    for model_type in model_types:

        sts_model = STSModel(
            model_type["name"],
            model_type["model"](),
            model_type["args"],
            input_names,
            train_n_test
        )
        sts_model_pool[dataset][sts_model.name] = sts_model
