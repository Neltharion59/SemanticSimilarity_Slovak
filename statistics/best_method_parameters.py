# Runnable script calculating statistic values to be printed to console.
# For each method determines which param configuration was the best of its configurations for how many datasets.

from evaluation.evaluate_regression_metrics import get_dataset_metrics, group_by_method, find_best_methods

import re
from os import listdir
from os.path import isfile, join
from functools import reduce
import operator as op

# Prepare path to read values from
input_path = "./../resources/datasets/sts_processed/"
# Prepare regexes to be used in this script
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?.txt")
# Prepare list of dataset files to be calculated
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]
# Prepare list of best methods (every method that is best for at least one dataset)
best_methods = map(lambda x: find_best_methods(group_by_method(get_dataset_metrics(x, print_2_screen=False))), input_dataset_files)
best_methods = reduce(op.add, best_methods)

result_dict = {}
# Loop over all best methods
for method_name in best_methods:
    # Retrieve method's name and param configuration
    tokens = method_name.split("___")
    method = tokens[0]
    params = tokens[1]
    # Create entries for method and params if they don't exist
    if method not in result_dict:
        result_dict[method] = {}
    if params not in result_dict[method]:
        result_dict[method][params] = 0
    # Increase the counter for current best method
    result_dict[method][params] = result_dict[method][params] + 1

# Loop over methods and print which of their configs was best how many times
for method_name in result_dict:
    print(method_name)
    # Loop over all param configs that were best at at least one dataset among this method's param configurations
    for params in sorted(result_dict[method_name].keys(), key=lambda x: result_dict[method_name][x], reverse=True):
        # Print results
        print("\t{}: {}".format(params, result_dict[method_name][params]))
