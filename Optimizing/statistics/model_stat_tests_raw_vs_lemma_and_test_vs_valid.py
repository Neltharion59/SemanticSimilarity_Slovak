# Runnable script calculating statistic values to be printed to console in CSV format.
# Perform statistical tests to detect significant differences of various categories of models.

import json

import numpy as np
from joblib import load
from scipy.stats import shapiro, ttest_ind, wilcoxon

from dataset_modification_scripts.dataset_pool import dataset_pool
from evaluation.evaluate_regression_metrics import pearson

alpha = 0.05
p_value_digit_count = 6
optimizer_run_ids = {
    '1st': [25, 33, 31, 30, 38, 41],
    '2nd': [62, 73, 70, 79, 76, 78]
}


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

all_values = {
    "valid": {},
    "test": {}
}

for dataset_version in ['raw', 'lemma']:

    values_valid = {}
    values_test = {}
    method_names = []

    for optimizer_run_name in optimizer_run_ids:

        path_to_model_template = paths[optimizer_run_name]['models']

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

                # Validation

                pearson_value_valid = pearson(trained_model.predict(features), labels)

                if model_type_name not in values_valid:
                    values_valid[model_type_name] = {}
                if dataset.name not in values_valid[model_type_name]:
                    values_valid[model_type_name][dataset.name] = {}

                values_valid[model_type_name][dataset.name][optimizer_run_name] = pearson_value_valid

                # Test
                trained_model = load(path_to_model_template.format(model_id, 'jolib'))
                with open(path_to_model_template.format(model_id, 'json'), 'r', encoding='utf-8') as file:
                    model_meta_dump = file.read()
                model_meta = json.loads(model_meta_dump)

                # Pearson on test set
                pearson_value_test = 1 - model_meta['fitness']

                if model_type_name not in values_test:
                    values_test[model_type_name] = {}
                if dataset.name not in values_test[model_type_name]:
                    values_test[model_type_name][dataset.name] = {}

                values_test[model_type_name][dataset.name][optimizer_run_name] = pearson_value_test

    all_values["valid"][dataset_version] = values_valid
    all_values["test"][dataset_version] = values_test

dataset_names = [dataset.name for dataset in dataset_pool['raw']]

# Raw vs. lemma
for run_name in optimizer_run_ids:
    for score_name in ["valid", "test"]:
        print('-' * 10 + ' {} - {} - Raw vs. Lema '.format(run_name, score_name) + '-' * 10)
        rows = [['Metóda', 'Test', 'p Hodnota', 'Významný rozdiel']]

        for model_type_name in all_values['valid']['raw']:
            row = []

            row.append(model_type_name)

            raw_values = [all_values[score_name]['raw'][model_type_name][dataset_name][run_name] for dataset_name in dataset_names]
            lema_values = [all_values[score_name]['lemma'][model_type_name][dataset_name + '_lemma'][run_name] for dataset_name in dataset_names]

            similarity_test_name = 't-test' if shapiro(raw_values).pvalue > alpha and shapiro(lema_values).pvalue > alpha else 'wilcox'
            row.append(similarity_test_name)

            similarity_test = (ttest_ind if similarity_test_name == 't-test' else wilcoxon)(raw_values, lema_values)
            row.append(round(similarity_test.pvalue, ndigits=p_value_digit_count))

            row.append('Áno' if similarity_test.pvalue < alpha else 'Nie')

            rows.append(row)

        for row in rows:
            print(','.join([str(x) for x in row]))

# Valid vs. Test
for run_name in optimizer_run_ids:
    for dataset_version in ["raw", "lemma"]:
        print('-' * 10 + ' {} - {} - Test vs. Valid '.format(run_name, dataset_version) + '-' * 10)
        rows = [['Metóda', 'Test', 'p Hodnota', 'Významný rozdiel']]

        for model_type_name in all_values['valid']['raw']:
            row = []

            row.append(model_type_name)

            test_values = [all_values['test'][dataset_version][model_type_name][dataset_name + ('' if dataset_version == 'raw' else '_lemma')][run_name] for dataset_name in
                          dataset_names]
            valid_values = [all_values['valid'][dataset_version][model_type_name][dataset_name + ('' if dataset_version == 'raw' else '_lemma')][run_name] for
                           dataset_name in dataset_names]

            similarity_test_name = 't-test' if shapiro(test_values).pvalue > alpha and shapiro(
                valid_values).pvalue > alpha else 'wilcox'
            row.append(similarity_test_name)

            similarity_test = (ttest_ind if similarity_test_name == 't-test' else wilcoxon)(test_values, valid_values)
            row.append(round(similarity_test.pvalue, ndigits=p_value_digit_count))

            row.append('Áno' if similarity_test.pvalue < alpha else 'Nie')

            rows.append(row)

        for row in rows:
            print(','.join([str(x) for x in row]))




