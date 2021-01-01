# Runnable script calculating metrics (MAE, MSE, PEARSON CORRELATION COEFFICIENT) for all datasets for all methods
# from persisted values and print them to screen. It also prints best configuration of parameters for each dataset
# for each method.

from model_management.sts_method_value_persistor import input_folder
from evaluation.evaluate_regression_metrics import get_dataset_metrics, group_by_method
from model_management.sts_method_value_persistor import gold_standard_name
import re
from os import listdir
from os.path import isfile, join

# Let's prepare regexes to be used throughout this script.
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

# Let's prepare list of files with values of methods to be evaluated.
input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]
input_dataset_files.sort()

# Delimiter for visual purposes
delimiter = "......................................................................................................" * 2

# Let's loop over all input dataset files
for dataset in input_dataset_files:

    # Let's print which dataset we are working with
    print(delimiter)
    print(dataset)

    # Retrieve metrics of the current dataset
    evaluation_values = get_dataset_metrics(dataset, print_2_screen=True)
    if evaluation_values is None:
        continue

    # We have metrics of all methods. But we consider each configuration of params of each method a separate method.
    # Let's group them by methods then.
    grouped_values = group_by_method(evaluation_values)

    # Let's loop over each method
    for method_group_name in grouped_values:
        # Calculating metrics of gold standard does not really make sense, so let's not do it.
        if method_group_name == gold_standard_name:
            continue
        print("Best values for {}".format(method_group_name))

        # Let's prepare the values of current method
        evaluation_values = grouped_values[method_group_name]

        # Let's determine which configurations of current method have best metrics.
        best_mae = min(evaluation_values, key=lambda x: evaluation_values[x][0])
        best_mse = min(evaluation_values, key=lambda x: evaluation_values[x][1])
        best_pearson = max(evaluation_values, key=lambda x: evaluation_values[x][2])

        # Let's also retrieve values of those metrics for those methods
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

        # Print the results
        print("Best MAE: {} - {}\nBest MSE: {} - {}\nBest Pearson: {} - {}".format(
            result_dict['MAE']['value'], result_dict['MAE']['name'],
            result_dict['MSE']['value'], result_dict['MSE']['name'],
            result_dict['PEARSON']['value'][0], result_dict['PEARSON']['name']
        ))
