# Library-like script providing functions for evaluating method metrics

from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.stats import pearsonr
from tabulate import tabulate
import re

# Let's prepare regexes to be used throughout this script.
model_method_name_regex = re.compile("model_[1-9]?[0-9]*")


# Finds best param configuration for each method and return list of those method names.
# Useful to determine which methods to feed aggregating model with.
# Params: dict
# Return: list<string>
def find_best_methods(grouped_method_results):
    # Initialize list of results
    results = []

    # Let's loop over all methods and choose the best param config for it
    for method_group_name in grouped_method_results:
        # We do not want to choose gold standard nor other aggregation models.
        if method_group_name == gold_standard_name or model_method_name_regex.match(method_group_name):
            continue

        # Let's prepare the values of current method
        evaluation_values = grouped_method_results[method_group_name]

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

        # Let's determine in how many metric is which configuration the best amongst all
        cardinality_dict = {}
        for metric in result_dict:
            if result_dict[metric]["name"] not in cardinality_dict:
                cardinality_dict[result_dict[metric]["name"]] = 0
            cardinality_dict[result_dict[metric]["name"]] = cardinality_dict[result_dict[metric]["name"]] + 1
        best_method = max(cardinality_dict.keys(), key=lambda x: cardinality_dict[x])

        # Let's append the name of best method to result list
        results.append(method_group_name + "___" + best_method)

    return results


# Group results of STS methods by method - because on input, same method with defferent param config
# is considered a separate method.
# Params: list<string, float...>
# Return: dict
def group_by_method(method_metrics):
    # Initialize the resulting dict
    result_dict = {}

    # Loop over all methods
    for method in method_metrics:
        # Let's retrieve the group name
        method_name = method[0].split("___")[0]
        # If method is a trained model, it does not contain param info.
        if model_method_name_regex.match(method_name):
            parameters_name = ""
        # Method is not a trained model, so we can retrieve param config, unless it's gold standard.
        else:
            parameters_name = "" if method_name == gold_standard_name else method[0].split("___")[1]
        # Add method name to dict if it is not already there
        if method_name not in result_dict:
            result_dict[method_name] = {}
        # Let's add config results to resulting dict
        result_dict[method_name][parameters_name] = method[1:]

    return result_dict


# Get metrics of given dataset based on persisted values.
# Params: str, [bool]
# Return: dict
def get_dataset_metrics(dataset_name, print_2_screen=False):
    # Let's get metrics of the dataset
    dataset_metrics = evaluate_dataset_metrics(dataset_name)

    # If something went wrong, there is not much left to do
    if dataset_metrics is None:
        return

    # Let's turn the metrics into a table that can be printed pretty
    tabulable_values = []
    for method_name in dataset_metrics:
        row = [method_name]
        for metric in dataset_metrics[method_name]:
            row.append(dataset_metrics[method_name][metric])
        tabulable_values.append(row)

    # If we are supposed to print the table to screen, let's do it.
    if print_2_screen:
        headers = ['STS Method']
        for metric in dataset_metrics[list(dataset_metrics.keys())[0]]:
            headers.append(metric)
        print(tabulate(tabulable_values, headers=headers))

    return tabulable_values


# Evaluate metrics of given dataset based on persisted values.
# Params: str
# Return: dict
def evaluate_dataset_metrics(dataset_name):
    # Get persisted values of all methods for given dataset
    method_values = get_persisted_method_values(dataset_name)

    # If we loaded nothing, there's nothing to do
    if method_values is None:
        print("Could not find any values for dataset {}".format(dataset_name))
        return

    # If there is no gold standard loaded, we cannot calculate enything either
    if gold_standard_name not in method_values:
        print("No gold standard found for dataset {}. Nothing to evaluate towards.".format(dataset_name))
        return

    # Let's prepare values of gold standard - we're going to need them
    gold_standard_values = method_values[gold_standard_name]

    model_metrics = {}

    # Loop over all methods and calculate their metrics
    for method_name in method_values:
        # If we're evaluating gold standard, we need to rescale it
        if method_name == gold_standard_name:
            model_metrics[method_name] = evaluate_prediction_metrics(gold_standard_values, method_values[method_name], 1)
        # Otherwise, just calculate metrics
        else:
            model_metrics[method_name] = evaluate_prediction_metrics(gold_standard_values, method_values[method_name])

    return model_metrics


# Calculate metrics from two vectors of values. Rescale, if needed.
# Params: list<float>, list<float>, [int, float]
# Return: dict
def evaluate_prediction_metrics(gold_standard_values, prediction_values, scaling_coefficient=5):
    # Rescale predictions
    prediction_values = list(map(lambda x: x * scaling_coefficient, prediction_values))
    # Calculate metrics from the values
    prediction_metrics = {
        'MAE': mean_absolute_error(gold_standard_values, prediction_values),
        'MSE': mean_squared_error(gold_standard_values, prediction_values),
        'PEARSON': pearsonr(gold_standard_values, prediction_values)
    }

    return prediction_metrics


def pearson(labels, predictions):
    return pearsonr(labels, predictions)[0]