import json
import math

import numpy as np
from joblib import load

from dataset_modification_scripts.dataset_pool import dataset_pool
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from evaluation.evaluate_regression_metrics import pearson
from model_management.sts_method_pool import character_based_name_list, corpus_based_name_list, \
    knowledge_based_name_list, term_based_name_list

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

R_code = 'library(ggplot2)\n'

max_values = {}

for dataset_version in ['raw', 'lemma']:

    R_code = R_code + '\n# --- {} ---\n'.format(dataset_version)

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
                    values[model_type_name][dataset.name] = []
                print(model_type_name, dataset.name)
                values[model_type_name][dataset.name].append(pearson_value)

    for model_name in values:
        for dataset_name in values[model_name]:
            values[model_name][dataset_name] = max(values[model_name][dataset_name])

    groups = [character_based_name_list, term_based_name_list, corpus_based_name_list, knowledge_based_name_list]
    group_names = ['Character-based', 'Term-based', 'Corpus-based', 'Knowledge-based']

    for dataset in dataset_pool[dataset_version]:
        persisted_values = dataset.load_values()

        for group, group_name in zip(groups, group_names):
            if group_name not in values:
                values[group_name] = {}

            values[group_name][dataset.name] = 0

            for method_name in group:
                values[group_name][dataset.name] = max([values[group_name][dataset.name]]
                    + [pearson(persisted_values[gold_standard_name][0]['values'], persisted_values[method_name][i]['values']) for i in range(len(persisted_values[method_name]))])

    value_vector = []
    name_vector = []
    for method_name in values:
        for dataset_name in values['linear_regression']:
            value_vector.append(values[method_name][dataset_name])
        name_vector.append(method_name)

    R_code = R_code\
        + 'df <- data.frame('\
            + ','.join([
                'x=rep(1:{}, {})'.format(len(dataset_pool['raw']), len(values.keys())),
                'val=c(' + ','.join([str(x) for x in value_vector]) + ')',
                'variable=rep(c({}), each={})'.format(
                    ','.join(['"' + x + '"' for x in name_vector]),
                    len(dataset_pool['raw'])
                )
            ])\
        + ')\n'

    R_code = R_code + 'ggplot(data = df, aes(x=x, y=val)) + geom_line(aes(colour=variable))'\
        + ' + scale_y_continuous(name = "Pearsonov koeficient nad validačnou množinou", limits=c(0, 1))'\
        + ' + scale_x_continuous(name=\'Dataset\', breaks=1:{}, labels=c('.format(len(dataset_pool['raw']))\
        + ','.join(['"{}"'.format(x.name.replace('semeval', 'se').replace('_lemma', '')) for x in dataset_pool['raw']])\
        + '))'

    for dataset_name in [dataset.name for dataset in dataset_pool[dataset_version]]:
        dataset_name_temp = dataset_name.replace('_lemma', '')
        if dataset_name_temp not in max_values:
            max_values[dataset_name_temp] = 0

        for method_name in values:
            max_values[dataset_name_temp] = max(max_values[dataset_name_temp], values[method_name][dataset_name])

print('Max_values: ', max_values)

print(R_code)




