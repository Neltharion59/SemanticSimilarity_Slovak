# Runnable script calculating statistic values to be printed to console in R code format.
# For given dataset version plots values of all algorithmic STS values in single plots.

import math

from util.math import average
from dataset_modification_scripts.dataset_pool import dataset_pool
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from evaluation.evaluate_regression_metrics import pearson
from model_management.sts_method_pool import character_based_name_list, corpus_based_name_list, \
    knowledge_based_name_list, term_based_name_list


# Put values into single vector.
# Params: dict<str, any>, list<str>
# Return: list<float>
def list_all_values(value_dict, method_name_list):
    all_values = []
    for method_name in method_name_list:
        all_values = all_values + value_dict[method_name]

    return all_values


key = 'lemma'

values = {}

for dataset in dataset_pool[key]:

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

all_method_names = []
all_dataset_names = [dataset.name for dataset in dataset_pool[key]]

for method_name in values:
    all_method_names.append(method_name)

dataset_count = len(all_dataset_names)

print('library(ggplot2)')

groups = [character_based_name_list, term_based_name_list, corpus_based_name_list, knowledge_based_name_list]
group_names = ['Character-based', 'Term-based', 'Corpus-based', 'Knowledge-based']

for group, group_name in zip(groups, group_names):
    print()

    print(
        'df <- data.frame('
        + ','.join([
            'x=rep(1:{}, {})'.format(dataset_count, len(group)),
            'val=c(' + ','.join([str(x) for x in list_all_values(values, group)]) + ')',
            'variable=rep(c({}), each={})'.format(
                ','.join(['"' + x + '"' for x in group]),
                dataset_count
            )
        ])
        + ')'
    )
    print(
        'ggplot(data = df, aes(x=x, y=val)) + geom_line(aes(colour=variable))' +
        ' + scale_y_continuous(name = "Najvyšší pearsonov koeficient", limits=c(0, 1))' +
        ' + scale_x_continuous(name=\'Dataset\', breaks=1:{}, labels=c('.format(dataset_count)
        + ','.join(['"{}"'.format(x.replace('semeval', 'se').replace('_lemma', '')) for x in all_dataset_names])
        + '))'
    )

print()

print(
    'df <- data.frame('
    + ','.join([
        'x=rep(1:{}, {})'.format(dataset_count, len(groups)),
        'val=c(' + ','.join([str(average([values[method][i] for method in group])) for group in groups for i in range(dataset_count)]) + ')',
        'variable=rep(c({}), each={})'.format(
            ','.join(['"' + x + '"' for x in group_names]),
            dataset_count
        )
    ])
    + ')'
)
print(
    'ggplot(data = df, aes(x=x, y=val)) + geom_line(aes(colour=variable))' +
    ' + scale_y_continuous(name = "Najvyšší pearsonov koeficient", limits=c(0, 1))' +
    ' + scale_x_continuous(name=\'Dataset\', breaks=1:{}, labels=c('.format(dataset_count)
    + ','.join(['"{}"'.format(x.replace('semeval', 'se').replace('_lemma', '')) for x in all_dataset_names])
    + '))'
)
