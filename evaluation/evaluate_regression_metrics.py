from model_management.sts_method_value_persistor import get_persisted_method_values, gold_standard_name
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.stats import pearsonr
from tabulate import tabulate


def print_dataset_metrics(dataset_name):
    dataset_metrics = evaluate_dataset_metrics(dataset_name)
    if dataset_metrics is None:
        return

    tabulable_values = []
    for method_name in dataset_metrics:
        row = [method_name]
        for metric in dataset_metrics[method_name]:
            row.append(dataset_metrics[method_name][metric])
        tabulable_values.append(row)
    headers = ['STS Method']
    for metric in dataset_metrics[list(dataset_metrics.keys())[0]]:
        headers.append(metric)
    print(tabulate(tabulable_values, headers=headers))


def evaluate_dataset_metrics(dataset_name):
    method_values = get_persisted_method_values(dataset_name)
    if method_values is None:
        print("Could not find any values for dataset {}".format(dataset_name))
        return

    if gold_standard_name not in method_values:
        print("No gold standard found for dataset {}. Nothing to evaluate towards.".format(dataset_name))
        return

    gold_standard_values = method_values[gold_standard_name]
    model_metrics = {}
    for method_name in method_values:
        model_metrics[method_name] = evaluate_prediction_metrics(gold_standard_values, method_values[method_name])
    return model_metrics


def evaluate_prediction_metrics(gold_standard_values, prediction_values):
    prediction_metrics = {
        'MAE': mean_absolute_error(gold_standard_values, prediction_values),
        'MSE': mean_squared_error(gold_standard_values, prediction_values),
        'PEARSON': pearsonr(gold_standard_values, prediction_values)
    }
    return prediction_metrics
