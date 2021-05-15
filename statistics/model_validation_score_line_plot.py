# Runnable script calculating statistic values to be printed to console in R code.
# For given run ids (sets of models of same type), generate R code, that will create plots of values
# across datasets for different model types in single plot.

import json

import numpy as np
from joblib import load

from dataset_modification_scripts.dataset_pool import dataset_pool
from evaluation.evaluate_regression_metrics import pearson

optimizer_run_ids = [62, 73, 70, 79, 76, 78]


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

R_code = 'library(ggplot2)\n'

for dataset_version in ['raw', 'lemma']:

    R_code = R_code + '\n# --- {} ---\n'.format(dataset_version)

    values = []
    model_type_names = []

    for optimizer_run_id in optimizer_run_ids:

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
        model_type_names.append(model_type_name)

        for dataset in dataset_pool[dataset_version]:

            optimizer_object = run_record['main'][dataset_version][dataset.name]['models'][model_type_name]
            # Load Model and its metadata
            model_id = optimizer_object['best_model_id']
            trained_model = load(path_to_model_template.format(model_id, 'jolib'))
            with open(path_to_model_template.format(model_id, 'json'), 'r', encoding='utf-8') as file:
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
            values.append(pearson_value)


    R_code = R_code\
        + 'df <- data.frame('\
            + ','.join([
                'x=rep(1:{}, {})'.format(len(dataset_pool['raw']), len(optimizer_run_ids)),
                'val=c(' + ','.join([str(x) for x in values]) + ')',
                'variable=rep(c({}), each={})'.format(
                    ','.join(['"' + x + '"' for x in model_type_names]),
                    len(dataset_pool['raw'])
                )
            ])\
        + ')\n'

    R_code = R_code + 'ggplot(data = df, aes(x=x, y=val)) + geom_line(aes(colour=variable))'\
        + ' + scale_y_continuous(name = "Pearsonov koeficient nad validačnou množinou", limits=c(0, 1))'\
        + ' + scale_x_continuous(name=\'Dataset\', breaks=1:{}, labels=c('.format(len(dataset_pool['raw']))\
        + ','.join(['"{}"'.format(x.name.replace('semeval', 'se').replace('_lemma', '')) for x in dataset_pool['raw']])\
        + '))'


print(R_code)




