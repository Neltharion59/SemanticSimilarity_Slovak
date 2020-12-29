import re
from os import listdir
from os.path import isfile, join
from functools import reduce

from evaluation.evaluate_regression_metrics import find_best_methods, group_by_method, get_dataset_metrics
from model_management.model_persistence import get_model_description, get_model_id_by_name, get_model_test_metrics
from statistics.semeval_pearson import semeval_person_results


def average_(iterable):
    count = 0
    sum_ = 0
    for item in iterable:
        if not isinstance(item, str):
            sum_ = sum_ + item
            count = count + 1
    return 0 if count == 0 else sum_/count


def max_(iterable):
    iterable = filter(lambda x: not isinstance(x, str), iterable)
    return max(iterable)


def median_(iterable):
    iterable = filter(lambda x: not isinstance(x, str), iterable)
    iterable = list(iterable)
    iterable.sort()
    middles = [iterable[int(len(iterable)/2)]] if len(iterable)%2 == 1 else [iterable[int(len(iterable)/2)], iterable[int(len(iterable)/2) + 1]]
    return average_(middles)


def mode_(iterable):
    iterable = filter(lambda x: not isinstance(x, str), iterable)
    card_dict = {}
    for item in iterable:
        if item not in card_dict:
            card_dict[item] = 0
        card_dict[item] = card_dict[item] + 1
    max_card = max(card_dict.values())
    results = []
    for item in card_dict:
        if card_dict[item] == max_card:
            results.append(item)
    results.sort()
    return results


debug_print = False

input_path = "./../resources/datasets/sts_processed/"

dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?.txt")
sts_dataset_name_pattern = re.compile("sts_201[2-6]_[a-zA-Z]+]")

input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

best_methods = map(lambda x: (x, find_best_methods(group_by_method(get_dataset_metrics(x, print_2_screen=False)))), input_dataset_files)
temp_best_method_dict = {}
for record in best_methods:
    temp_best_method_dict[record[0]] = record[1]

best_method_dict = {}
for dataset_name in temp_best_method_dict:
    group_name = dataset_name.replace("_sk.txt", "").replace("_sk_lemma.txt", "")
    if group_name not in best_method_dict:
        best_method_dict[group_name] = {}
    best_method_dict[group_name][dataset_name] = {}
    best_method_dict[group_name][dataset_name]["best_methods"] = temp_best_method_dict[dataset_name]

if debug_print:
    print("Best method dict")
    for x in best_method_dict:
        print(x)
        for y in best_method_dict[x]:
            print("\t{}".format(y))
            for z in best_method_dict[x][y]:
                print("\t\t{}: {}".format(z, best_method_dict[x][y][z]))

all_method_metrics = map(lambda x: (x, get_dataset_metrics(x, print_2_screen=False)), input_dataset_files)
all_method_metrics = list(all_method_metrics)

method_metric_dict = {}
for x in all_method_metrics:
    method_metric_dict[x[0]] = {}
    for method in x[1]:
        method_name = method[0]
        method_metric_dict[x[0]][method_name] = {}
        method_metric_dict[x[0]][method_name]["MAE"] = method[1]
        method_metric_dict[x[0]][method_name]["MSE"] = method[2]
        method_metric_dict[x[0]][method_name]["PEARSON"] = method[3][0]

if debug_print:
    print("All results - {}:".format(len(all_method_metrics)))
    for dataset in method_metric_dict:
        print(dataset)
        for method in method_metric_dict[dataset]:
            print("\t{}".format(method))
            for metric in method_metric_dict[dataset][method]:
                print("\t\t{}: {}".format(metric, method_metric_dict[dataset][method][metric]))

model_regex = re.compile("model_[0-9]+")
result_table = []
for dataset_group in best_method_dict:

    dataset_names = [dataset_group + "_sk.txt", dataset_group + "_sk_lemma.txt"]
    for dataset_name in dataset_names:
        best_methods = {
            best_method.split("___")[0]: best_method for best_method in best_method_dict[dataset_group][dataset_name]["best_methods"]
        }

        result_row = [dataset_name.replace("_sk", "").replace(".txt", "")]
        result_row.append(method_metric_dict[dataset_name][best_methods["hamming"]]["PEARSON"])
        result_row.append(method_metric_dict[dataset_name][best_methods["wu_palmer"]]["PEARSON"])
        result_row.append(method_metric_dict[dataset_name][best_methods["path"]]["PEARSON"])
        result_row.append(method_metric_dict[dataset_name][best_methods["leacock_chodorow"]]["PEARSON"])

        models = map(lambda x: (get_model_description(int(x.split("_")[1])).split("___")[0], x), filter(lambda x: model_regex.match(x), method_metric_dict[dataset_name].keys()))
        models = {x[0]: x[1] for x in models}

        result_row.append(method_metric_dict[dataset_name][models["linear_regression"]]["PEARSON"])
        result_row.append(method_metric_dict[dataset_name][models["support_vector_regression"]]["PEARSON"])
        result_row.append(method_metric_dict[dataset_name][models["decision_tree_regression"]]["PEARSON"])

        result_table.append(result_row)

# Round values to 2 places
result_table = [[row[0]] + [round(x, 2) for x in row[1:]] for row in result_table]

result_table = [row + ([round(semeval_person_results[row[0].replace("_lemma", "") + "_sk.txt"], 2)] if (row[0].replace("_lemma", "") + "_sk.txt") in semeval_person_results else ["N/A"]) for row in result_table]

for i in range(len(result_table)):
    result_table[i][0] = result_table[i][0].replace("dataset_", "")

# Add metrics of models on testing dataset
for row in result_table:
    dataset = ("dataset_" + row[0].replace("_lemma", "") + "_sk_lemma.txt") if "_lemma" in row[0] else ("dataset_" + row[0] + "_sk.txt")
    model_names = ["linear_regression___", "support_vector_regression___", "decision_tree_regression___"]
    starting_index = 5
    for model_name in model_names:
        model_id = get_model_id_by_name(dataset, model_name)
        model_params = get_model_test_metrics(model_id)
        pearson = round(model_params["PEARSON"], 2)
        row[starting_index] = "{} ({})".format(row[starting_index], pearson)

        starting_index = starting_index + 1

average_row = ["Average"]
max_row = ["Max"]
median_row = ["Median"]
mode_row = ["Mode"]
for i in range(1, len(result_table[0])):
    if i not in [5, 6, 7]:
        average_row.append(round(average_(map(lambda x: x[i], result_table)), 2))
        max_row.append(max_(map(lambda x: x[i], result_table)))
        median_row.append(round(median_(map(lambda x: x[i], result_table)), 2))
        mode_row.append(",".join(list(map(lambda x: str(round(x, 2)), mode_(map(lambda x: x[i], result_table))))))
    else:
        all_values = list(map(lambda x: float(x[i].split(" ")[0]), result_table))
        test_values = list(map(lambda x: float(x[i].split(" ")[1].replace("(", "").replace(")", "")), result_table))

        print(all_values)
        print(test_values)

        average_row.append("{} ({})".format(round(average_(all_values), 2), round(average_(test_values), 2)))
        max_row.append("{} ({})".format(round(max_(all_values), 2), round(max_(test_values), 2)))
        median_row.append("{} ({})".format(round(median_(all_values), 2), round(median_(test_values), 2)))
        mode_row.append("{} ({})".format(
            ",".join(list(map(lambda x: str(round(x, 2)), mode_(all_values)))),
            ",".join(list(map(lambda x: str(round(x, 2)), mode_(test_values))))
        ))

result_table.append(average_row)
result_table.append(mode_row)
result_table.append(median_row)
result_table.append(max_row)

for row in result_table:
    print("\t".join(map(str, row)))
