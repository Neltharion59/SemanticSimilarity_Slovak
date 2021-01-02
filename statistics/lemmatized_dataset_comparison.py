# Runnable script calculating statistic values to be printed to console.
# Generate data in table form that can be imported to MS Excel so that it can be included in DP2 report.
# For each dataset, we have Pearson coefficient(pc_) of all methods (simple methods with best param config)
# Regression models have value of Pearson coefficient on all data first and in bracets value on testing data second
# (testing data is more relevant, as it does not include potentially overfitted data).
# Row order: dataset_name, pc_hamming, pc_wu_palmer, pc_path, pc_leacock_chodorow, pc_linear_regression(pc_test),
#            pc_svm_regression(pc_test), pc_decision_tree(pc_test), pc_best_from_SemEval

import re
from os import listdir
from os.path import isfile, join

from evaluation.evaluate_regression_metrics import find_best_methods, group_by_method, get_dataset_metrics
from model_management.model_persistence import get_model_description, get_model_id_by_name, get_model_test_metrics
from statistics.semeval_pearson import semeval_person_results


# Simple handy function to calculate average of collection of numbers.
# Resistant to empty collections and removes strings from collection (first element is usually name of dataset).
# Params: list<float/int/str>
# Return: float/int
def average_(iterable):
    count = 0
    sum_ = 0
    for item in iterable:
        if not isinstance(item, str):
            sum_ = sum_ + item
            count = count + 1
    return 0 if count == 0 else sum_/count


# Simple handy function to calculate max of collection of numbers.
# Removes strings from collection (first element is usually name of dataset).
# Params: list<float/int/str>
# Return: float/int
def max_(iterable):
    iterable = filter(lambda x: not isinstance(x, str), iterable)
    return max(iterable)


# Simple handy function to calculate median of collection of numbers.
# Removes strings from collection (first element is usually name of dataset).
# Params: list<float/int/str>
# Return: float/int
def median_(iterable):
    iterable = filter(lambda x: not isinstance(x, str), iterable)
    iterable = list(iterable)
    iterable.sort()
    middles = [iterable[int(len(iterable)/2)]] if len(iterable)%2 == 1 else [iterable[int(len(iterable)/2)], iterable[int(len(iterable)/2) + 1]]
    return average_(middles)


# Simple handy function to calculate mode(modus) of collection of numbers.
# Removes strings from collection (first element is usually name of dataset).
# Params: list<float/int/str>
# Return: float/int
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


# Is this debug session so we want extra printing to console or not?
debug_print = False

# Prepare path to read data from
input_path = "./../resources/datasets/sts_processed/"

# Prepare regexes to be used in this script
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?.txt")
sts_dataset_name_pattern = re.compile("sts_201[2-6]_[a-zA-Z]+]")

# Prepare list of dataset files to read data from
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

# Get dict of best methods (best config of each method) for each dataset
best_methods = map(lambda x: (x, find_best_methods(group_by_method(get_dataset_metrics(x, print_2_screen=False)))), input_dataset_files)
temp_best_method_dict = {}
for record in best_methods:
    temp_best_method_dict[record[0]] = record[1]

# Organize the methods in handy dict
best_method_dict = {}
for dataset_name in temp_best_method_dict:
    group_name = dataset_name.replace("_sk.txt", "").replace("_sk_lemma.txt", "")
    if group_name not in best_method_dict:
        best_method_dict[group_name] = {}
    best_method_dict[group_name][dataset_name] = {}
    best_method_dict[group_name][dataset_name]["best_methods"] = temp_best_method_dict[dataset_name]

# Check how the dict was constructed if we are debugging
if debug_print:
    print("Best method dict")
    for x in best_method_dict:
        print(x)
        for y in best_method_dict[x]:
            print("\t{}".format(y))
            for z in best_method_dict[x][y]:
                print("\t\t{}: {}".format(z, best_method_dict[x][y][z]))

# Retrieve metrics for all relevant methods
all_method_metrics = map(lambda x: (x, get_dataset_metrics(x, print_2_screen=False)), input_dataset_files)
all_method_metrics = list(all_method_metrics)

# Put the metrics to handy dict
method_metric_dict = {}
for x in all_method_metrics:
    method_metric_dict[x[0]] = {}
    for method in x[1]:
        method_name = method[0]
        method_metric_dict[x[0]][method_name] = {}
        method_metric_dict[x[0]][method_name]["MAE"] = method[1]
        method_metric_dict[x[0]][method_name]["MSE"] = method[2]
        method_metric_dict[x[0]][method_name]["PEARSON"] = method[3][0]

# Check how the dict was constructed if we are debugging
if debug_print:
    print("All results - {}:".format(len(all_method_metrics)))
    for dataset in method_metric_dict:
        print(dataset)
        for method in method_metric_dict[dataset]:
            print("\t{}".format(method))
            for metric in method_metric_dict[dataset][method]:
                print("\t\t{}: {}".format(metric, method_metric_dict[dataset][method][metric]))

# Get metrics for models. We have mapping between model id and description, so it's a bit complicated.
# Let's not get into detail here.
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
# Modify dataset names to bare minimum, otherwise they will be too wide for A4 page
result_table = [row + ([round(semeval_person_results[row[0].replace("_lemma", "") + "_sk.txt"], 2)] if (row[0].replace("_lemma", "") + "_sk.txt") in semeval_person_results else ["N/A"]) for row in result_table]
for i in range(len(result_table)):
    result_table[i][0] = result_table[i][0].replace("dataset_", "")

# Add metrics of models on testing dataset (within brackets to corresponding columns)
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

# Group datasets by original datasets (in 1 group will be raw dataset and its lemmatized dataset)
dataset_dict = {}
for row in result_table:
    is_lemma = "_lemma" in row[0]
    dataset_name = row[0].replace("_lemma", "")
    if dataset_name not in dataset_dict:
        dataset_dict[dataset_name] = {}
    dataset_dict[dataset_name]["lemma" if is_lemma else "raw"] = row[1:-1]

# Create new table - dataset group in one row
new_table = []
for dataset_name in dataset_dict:
    row = [dataset_name]
    for i in range(len(dataset_dict[dataset_name]["lemma"])):
        row.append(dataset_dict[dataset_name]["raw"][i])
        row.append(dataset_dict[dataset_name]["lemma"][i])
    new_table.append(row)

result_table = new_table

# Below metrics of datasets, let's also calculate average, max, median, mode for each method
average_row = ["Average"]
max_row = ["Max"]
median_row = ["Median"]
mode_row = ["Mode"]
for i in range(1, len(result_table[0])):
    if i not in [9, 10, 11, 12, 13, 14]:
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

# Split the table to 2 tables -> simple methods and aggregation methods
result_table_methods_simple = list(map(lambda x: x[:9], result_table))
result_table_methods_complex = list(map(lambda x: [x[0]] + x[9:], result_table))
visual_delimiter = "." * 100

print(visual_delimiter)
for row in result_table_methods_simple:
    print("\t".join(map(str, row)))
print(visual_delimiter)
for row in result_table_methods_complex:
    print("\t".join(map(str, row)))
print(visual_delimiter)
