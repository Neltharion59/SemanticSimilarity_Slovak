from evaluation.evaluate_regression_metrics import get_dataset_metrics, group_by_method, find_best_methods

import re
from os import listdir
from os.path import isfile, join
from functools import reduce
import operator as op

input_path = "./../resources/datasets/sts_processed/"

dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?.txt")

input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]


best_methods = map(lambda x: find_best_methods(group_by_method(get_dataset_metrics(x, print_2_screen=False))), input_dataset_files)
best_methods = reduce(op.add, best_methods)

result_dict = {}
for method_name in best_methods:
    tokens = method_name.split("___")
    method = tokens[0]
    params = tokens[1]
    if method not in result_dict:
        result_dict[method] = {}
    if params not in result_dict[method]:
        result_dict[method][params] = 0
    result_dict[method][params] = result_dict[method][params] + 1

for method_name in result_dict:
    print(method_name)
    for params in sorted(result_dict[method_name].keys(), key=lambda x: result_dict[method_name][x], reverse=True):
        print("\t{}: {}".format(params, result_dict[method_name][params]))
