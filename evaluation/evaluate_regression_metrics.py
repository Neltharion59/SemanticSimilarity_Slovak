from model_management.sts_method_value_persistor import get_persisted_method_values, gold_standard_name
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.stats import pearsonr
from tabulate import tabulate


def find_best_methods(grouped_method_results):
    results = []

    for method_group_name in grouped_method_results:
        if method_group_name == gold_standard_name:
            continue

        evaluation_values = grouped_method_results[method_group_name]

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

        cardinality_dict = {}
        for metric in result_dict:
            if result_dict[metric]["name"] not in cardinality_dict:
                cardinality_dict[result_dict[metric]["name"]] = 0
            cardinality_dict[result_dict[metric]["name"]] = cardinality_dict[result_dict[metric]["name"]] + 1
        best_method = max(cardinality_dict.keys(), key=lambda x: cardinality_dict[x])
        results.append(method_group_name + "___" + best_method)

    return results


def group_by_method(method_metrics):
    result_dict = {}
    for method in method_metrics:
        method_name = method[0].split("___")[0]
        parameters_name = "" if method_name == gold_standard_name else method[0].split("___")[1]
        if method_name not in result_dict:
            result_dict[method_name] = {}
        result_dict[method_name][parameters_name] = method[1:]

    return result_dict


def get_dataset_metrics(dataset_name, print_2_screen=False):
    dataset_metrics = evaluate_dataset_metrics(dataset_name)
    if dataset_metrics is None:
        return

    tabulable_values = []
    for method_name in dataset_metrics:
        row = [method_name]
        for metric in dataset_metrics[method_name]:
            row.append(dataset_metrics[method_name][metric])
        tabulable_values.append(row)
    if print_2_screen:
        headers = ['STS Method']
        for metric in dataset_metrics[list(dataset_metrics.keys())[0]]:
            headers.append(metric)
        print(tabulate(tabulable_values, headers=headers))

    return tabulable_values


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
        if method_name == gold_standard_name:
            model_metrics[method_name] = evaluate_prediction_metrics(gold_standard_values, method_values[method_name], 1)
        else:
            model_metrics[method_name] = evaluate_prediction_metrics(gold_standard_values, method_values[method_name])
    return model_metrics


def evaluate_prediction_metrics(gold_standard_values, prediction_values, scaling_coefficient=5):
    prediction_values = list(map(lambda x: x * scaling_coefficient, prediction_values))
    prediction_metrics = {
        'MAE': mean_absolute_error(gold_standard_values, prediction_values),
        'MSE': mean_squared_error(gold_standard_values, prediction_values),
        'PEARSON': pearsonr(gold_standard_values, prediction_values)
    }
    return prediction_metrics
