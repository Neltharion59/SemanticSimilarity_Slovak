from model_management.sts_method_value_persistor import input_folder
from evaluation.evaluate_regression_metrics import get_dataset_metrics, group_by_method
from model_management.sts_method_value_persistor import gold_standard_name
import re
from os import listdir
from os.path import isfile, join


dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]
input_dataset_files.sort()

delimiter = "......................................................................................................" * 2

for dataset in input_dataset_files:
    print(delimiter)
    print(dataset)

    evaluation_values = get_dataset_metrics(dataset, print_2_screen=True)
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
                'name': best_mse,
                'value': evaluation_values[best_mse][1]
            },
            'PEARSON': {
                'name': best_pearson,
                'value': evaluation_values[best_pearson][2]
            }
        }

        print("Best MAE: {} - {}\nBest MSE: {} - {}\nBest Pearson: {} - {}".format(
            result_dict['MAE']['value'], result_dict['MAE']['name'],
            result_dict['MSE']['value'], result_dict['MSE']['name'],
            result_dict['PEARSON']['value'][0], result_dict['PEARSON']['name']
        ))
