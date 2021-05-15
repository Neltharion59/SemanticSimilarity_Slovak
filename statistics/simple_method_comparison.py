# Runnable script calculating statistic values to be printed to console in CSV format.
# Compares values of algorithmic STS methods.

import math

from util.math import average
from dataset_modification_scripts.dataset_pool import dataset_pool
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from evaluation.evaluate_regression_metrics import pearson


def format_number(stringed_number):
    if len(stringed_number.split('.')[1]) == 1:
        return stringed_number + '0'
    return stringed_number


values = {}

header_row = ['Metóda']

for dataset in dataset_pool['lemma']:
    header_row.append(dataset.name.replace('semeval', 'se').replace('_lemma', ''))

    persisted_values = dataset.load_values()

    for method_name in persisted_values:

        if method_name == gold_standard_name:
            continue

        pearson_values = []
        for record in persisted_values[method_name]:
            pearson_value = pearson([x/5 for x in persisted_values['gold_standard'][0]['values']], record['values'])
            if not math.isnan(pearson_value):
                pearson_values.append(pearson_value)

        aggregated_pearson = max(pearson_values)

        if method_name not in values:
            values[method_name] = []

        values[method_name].append(aggregated_pearson)

print(','.join(header_row))
for method_name in values:
    print(','.join([method_name] + [format_number(str(round(x, ndigits=2))) for x in values[method_name]]))

max_row = []
for i in range(1, len(header_row)):
    max_row.append(format_number(str(round(max([values[method_name][i - 1] for method_name in values]), ndigits=2))))
print(','.join(['Maximum'] + max_row))

avg_row = []
for i in range(1, len(header_row)):
    avg_row.append(format_number(str(round(average([values[method_name][i - 1] for method_name in values]), ndigits=2))))
print(','.join(['Priemer'] + avg_row))

min_row = []
for i in range(1, len(header_row)):
    min_row.append(format_number(str(round(min([values[method_name][i - 1] for method_name in values]), ndigits=2))))
print(','.join(['Minimum'] + min_row))
