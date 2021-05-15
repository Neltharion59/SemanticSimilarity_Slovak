# Runnable script calculating statistic values to be printed to console in CSV format.
# For given run id (set of models of same type),
# compares various metrics (on test set, on validation set, on semeval-all dataset, on all dataset)
# across datasets and dataset types.

import json

import numpy as np
from joblib import load

from util.math import average
from dataset_modification_scripts.dataset_pool import dataset_pool, find_dataset_by_name
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from evaluation.evaluate_regression_metrics import pearson

optimizer_run_id = 62


# Get index of given configuration in list of configurations of given method
# Params: dict<str, any>, list<dict<str, any>>
# Return: int
def configuration_index(config, config_list):

    for i in range(len(config_list)):
        fail = False

        for key in config:
            if key not in config_list[i] or config_list[i][key] != config[key]:
                fail = True
                break

        if fail:
            continue

        for key in config_list[i]:
            if key not in config or config_list[i][key] != config[key]:
                fail = True
                break

        if fail:
            continue

        return i

    return -1


# Transposes array.
# Params: list<list<float>>
# Return: list<list<float>>
def transpose(array):
    return np.array(array).T.tolist()


# Formats number in string form to have two digits (if it only had one)
# Params: str
# Return: str
def format_number(stringed_number):
    if len(stringed_number.split('.')[1]) == 1:
        return stringed_number + '0'
    return stringed_number


path_to_optimizer_run_record_folder = './../resources/optimizer_runs/'
path_to_model_template = './../resources/models/model_{}.{}'

try:
    with open(
            '{}optimizer_run_{}.txt'.format(path_to_optimizer_run_record_folder, optimizer_run_id),
            'r',
            encoding='utf-8'
    ) as file:
        txt_dump = file.read()
    print('--- OLD --- Run with id {} exists.\n --- START ---'.format(optimizer_run_id))
    run_record = json.loads(txt_dump)
except FileNotFoundError:
    print('--- NEW --- Run with id {} does not exist.\n --- OVER ---'.format(optimizer_run_id))
    exit(1)

model_type_name = list(run_record['main']['lemma']['all_lemma']['models'].keys())[0]
print(model_type_name)

rows = [
    ['', 'Raw', '', '', '', 'Lema', '', '', ''],
    [''] + ['Test', 'Valid', 'se-all', 'all'] * 2
]
for dataset in dataset_pool['raw']:
    row = [dataset.name.replace('semeval', 'se')]

    for dataset_version in ['raw', 'lemma']:

        optimizer_object = run_record['main'][dataset_version][dataset.name if dataset_version == 'raw' else (dataset.name + "_lemma")]['models'][model_type_name]

        model_id = optimizer_object['best_model_id']
        trained_model = load(path_to_model_template.format(model_id, 'jolib'))
        with open(path_to_model_template.format(model_id, 'json'), 'r', encoding='utf-8') as file:
            model_meta_dump = file.read()
        model_meta = json.loads(model_meta_dump)

        # Pearson on test set
        row.append(1 - model_meta['fitness'])

        inputs = model_meta['inputs']

        # Pearson on valid. test
        features = []
        for method in inputs:
            index = configuration_index(method['args'], [x['args'] for x in model_meta['validation']['features'][method['method_name']]])
            if index == -1:
                raise ValueError('Args not found')
            features.append(
                model_meta['validation']['features'][method['method_name']][index]['values']
            )
        features = transpose(features)

        labels = model_meta['validation']['labels']

        row.append(pearson(trained_model.predict(features), labels))

        # Pearson on semeval-all
        dataset_values = find_dataset_by_name(dataset_version, 'semeval-all' if dataset_version == 'raw' else "semeval-all_lemma").load_values()
        features = []
        for method in inputs:
            index = configuration_index(method['args'], [x['args'] for x in
                                                         dataset_values[method['method_name']]])
            if index == -1:
                raise ValueError('Args not found')
            features.append(
                dataset_values[method['method_name']][index]['values']
            )
        features = transpose(features)

        labels = dataset_values[gold_standard_name][0]['values']
        row.append(pearson(trained_model.predict(features), labels))

        # all
        dataset_values = find_dataset_by_name(dataset_version, 'all' if dataset_version == 'raw' else "all_lemma").load_values()
        features = []
        for method in inputs:
            index = configuration_index(method['args'], [x['args'] for x in
                                                         dataset_values[method['method_name']]])
            if index == -1:
                raise ValueError('Args not found')
            features.append(
                dataset_values[method['method_name']][index]['values']
            )
        features = transpose(features)

        labels = dataset_values[gold_standard_name][0]['values']
        row.append(pearson(trained_model.predict(features), labels))

    rows.append(row)

max_row, avg_row, min_row = ['Maximum'], ['Priemer'], ['Minimum']
for i in range(1, len(rows[0])):
    vector = [row[i] for row in rows[2:]]

    max_row.append(max(vector))
    avg_row.append(average(vector))
    min_row.append(min(vector))

rows.append(max_row)
rows.append(avg_row)
rows.append(min_row)

rows = [[format_number(str(round(x, ndigits=2))) if isinstance(x, float) else x for x in row] for row in rows]

for row in rows:
    print(','.join(row))
