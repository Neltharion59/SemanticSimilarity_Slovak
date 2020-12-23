from model_management.sts_method_value_persistor import input_folder
from evaluation.evaluate_regression_metrics import print_dataset_metrics
from model_management.sts_method_value_persistor import gold_standard_name
import re
from os import listdir
from os.path import isfile, join


def group_by_method(method_metrics):
    result_dict = {}
    for method in method_metrics:
        method_name = method[0].split("___")[0]
        parameters_name = "" if method_name == gold_standard_name else method[0].split("___")[1]
        if method_name not in result_dict:
            result_dict[method_name] = {}
        result_dict[method_name][parameters_name] = method[1:]

    return result_dict


dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]
input_dataset_files.sort()

delimiter = "......................................................................................................" * 2

for dataset in input_dataset_files:
    print(delimiter)
    print(dataset)

    evaluation_values = print_dataset_metrics(dataset)
    if evaluation_values is None:
        continue
        
    grouped_values = group_by_method(evaluation_values)

    for method_group_name in grouped_values:
        if method_group_name == gold_standard_name:
            continue
        print("Best values for {}".format(method_group_name))

        evaluation_values = grouped_values[method_group_name]

        best_mae = min(evaluation_values, key=lambda x: evaluation_values[x][0])
        best_mse = min(evaluation_values, key=lambda x: evaluation_values[x][1])
        best_pearson = max(evaluation_values, key=lambda x: evaluation_values[x][2])

        result_dict = {
            'MAE': {
                'name': best_mae,
                'value': evaluation_values[best_mae][0]
            },
            'MSE': {
                'name': best_mae,
                'value': evaluation_values[best_mae][1]
            },
            'PEARSON': {
                'name': best_mae,
                'value': evaluation_values[best_mae][2]
            }
        }

        print("Best MAE: {} - {}\nBest MSE: {} - {}\nBest Pearson: {} - {}".format(
            result_dict['MAE']['value'], result_dict['MAE']['name'],
            result_dict['MSE']['value'], result_dict['MSE']['name'],
            result_dict['PEARSON']['value'][0], result_dict['PEARSON']['name']
        ))
