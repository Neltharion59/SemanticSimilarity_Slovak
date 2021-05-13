import json
import math

import numpy as np
from joblib import load
from scipy.stats import shapiro, ttest_ind, wilcoxon

from dataset_modification_scripts.dataset_pool import dataset_pool
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from evaluation.evaluate_regression_metrics import pearson
from model_management.sts_method_pool import character_based_name_list, corpus_based_name_list, \
    knowledge_based_name_list, term_based_name_list

alpha = 0.05
p_value_digit_count = 6
optimizer_run_ids = {
    '1st': [25, 33, 31, 30, 38, 41],
    '2nd': [62, 73, 70, 79, 76, 78]
}


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


def transpose(array):
    return np.array(array).T.tolist()


def format_number(stringed_number):
    if len(stringed_number.split('.')[1]) == 1:
        return stringed_number + '0'
    return stringed_number


paths = {
    '1st': {
        'run_record': './../resources/optimizer_runs_1/',
        'models': './../resources/models_1/model_{}.{}'
    },
    '2nd': {
        'run_record': './../resources/optimizer_runs/',
        'models': './../resources/models/model_{}.{}'
    }
}

all_values = {}

for dataset_version in ['raw', 'lemma']:

    values = {}
    method_names = []

    for optimizer_run_name in optimizer_run_ids:
        for optimizer_run_id in optimizer_run_ids[optimizer_run_name]:

            try:
                with open(
                        '{}optimizer_run_{}.txt'.format(paths[optimizer_run_name]['run_record'], optimizer_run_id),
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
            method_names.append(model_type_name)

            for dataset in dataset_pool[dataset_version]:

                optimizer_object = run_record['main'][dataset_version][dataset.name]['models'][model_type_name]
                # Load Model and its metadata
                model_id = optimizer_object['best_model_id']
                trained_model = load(paths[optimizer_run_name]['models'].format(model_id, 'jolib'))
                with open(paths[optimizer_run_name]['models'].format(model_id, 'json'), 'r', encoding='utf-8') as file:
                    model_meta_dump = file.read()
                model_meta = json.loads(model_meta_dump)

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

                pearson_value = pearson(trained_model.predict(features), labels)

                if model_type_name not in values:
                    values[model_type_name] = {}
                if dataset.name not in values[model_type_name]:
                    values[model_type_name][dataset.name] = {}

                values[model_type_name][dataset.name][optimizer_run_name] = pearson_value

    all_values[dataset_version] = values

dataset_names = [dataset.name for dataset in dataset_pool['raw']]

# Raw vs. lemma
for run_name in optimizer_run_ids:
    print('-' * 10 + ' {} - Raw vs. Lema '.format(run_name) + '-' * 10)
    rows = [['Metóda', 'Test', 'p Hodnota', 'Významný rozdiel']]

    for model_type_name in all_values['raw']:
        row = []

        row.append(model_type_name)

        raw_values = [all_values['raw'][model_type_name][dataset_name][run_name] for dataset_name in dataset_names]
        lema_values = [all_values['lemma'][model_type_name][dataset_name + '_lemma'][run_name] for dataset_name in dataset_names]

        similarity_test_name = 't-test' if shapiro(raw_values).pvalue > alpha and shapiro(lema_values).pvalue > alpha else 'wilcox'
        row.append(similarity_test_name)

        similarity_test = (ttest_ind if similarity_test_name == 't-test' else wilcoxon)(raw_values, lema_values)
        row.append(round(similarity_test.pvalue, ndigits=p_value_digit_count))

        row.append('Áno' if similarity_test.pvalue < alpha else 'Nie')

        rows.append(row)

    for row in rows:
        print(','.join([str(x) for x in row]))

# 1st vs 2nd
for dataset_version in ['raw', 'lemma']:
    if dataset_version == 'lemma':
        dataset_names = [(dataset_name + "_lemma") for dataset_name in dataset_names]

    print('-' * 10 + ' {} - 1st vs. 2nd '.format(dataset_version) + '-' * 10)
    rows = [['Metóda', 'Test', 'p Hodnota', 'Významný rozdiel']]

    for model_type_name in all_values[dataset_version]:
        row = []

        row.append(model_type_name)

        first_values = [all_values[dataset_version][model_type_name][dataset_name]['1st'] for dataset_name in dataset_names]
        second_values = [all_values[dataset_version][model_type_name][dataset_name]['2nd'] for dataset_name in dataset_names]

        similarity_test_name = 't-test' if shapiro(first_values).pvalue > alpha and shapiro(second_values).pvalue > alpha else 'wilcox'
        row.append(similarity_test_name)

        similarity_test = (ttest_ind if similarity_test_name == 't-test' else wilcoxon)(first_values, second_values)
        row.append(round(similarity_test.pvalue, ndigits=p_value_digit_count))

        row.append('Áno' if similarity_test.pvalue < alpha else 'Nie')

        rows.append(row)

    for row in rows:
        print(','.join([str(x) for x in row]))




