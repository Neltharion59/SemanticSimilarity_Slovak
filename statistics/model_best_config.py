import json, numpy as np
from evaluation.evaluate_regression_metrics import pearson
from functools import reduce
from joblib import load


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


# Params: float
# Return: string
def trim_number(number, digits=3):
    return str(round(number, digits))


# Params: string
# Return: string
def abbreviate_model_type(model_type):
    if model_type == 'random_forest_regression':
        return 'RFR'
    elif model_type == 'linear_regression':
        return 'LR'
    elif model_type == 'support_vector_regression':
        return 'SVR'
    elif model_type == 'gradient_boosting_regression':
        return 'GBR'
    elif model_type == 'bayesan_ridge_regression':
        return 'BRR'
    elif model_type == 'decision_tree_regression':
        return 'DTR'
    else:
        print('Unknown model type:' + str(model_type))
        exit(1)


optimizer_run_ids = {
    '1st': [25, 33, 31, 30, 38, 41],
    '2nd': [62, 73, 70, 79, 76, 78]
}

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

output_table = {}

for optimizer_run_name in optimizer_run_ids:

    output_table[optimizer_run_name] = {}

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

        for dataset_version in ['raw', 'lemma']:

            if dataset_version not in output_table[optimizer_run_name]:
                output_table[optimizer_run_name][dataset_version] = {}

            for dataset_name in run_record['main'][dataset_version]:

                if dataset_name not in output_table[optimizer_run_name][dataset_version]:
                    output_table[optimizer_run_name][dataset_version][dataset_name] = {}

                if len(list(run_record['main'][dataset_version][dataset_name]['models'].values())) != 1:
                    print("Invalid format")
                    exit(1)

                current_model_id = list(run_record['main'][dataset_version][dataset_name]['models'].values())[0]['best_model_id']

                if 'best_model_ids' not in output_table[optimizer_run_name][dataset_version][dataset_name]:
                    output_table[optimizer_run_name][dataset_version][dataset_name] = {
                        'best_model_ids': []
                    }

                output_table[optimizer_run_name][dataset_version][dataset_name]['best_model_ids'].append(current_model_id)


for optimizer_run_name in output_table:
    for dataset_version in output_table[optimizer_run_name]:
        for dataset_name in output_table[optimizer_run_name][dataset_version]:

            models = []

            for model_id in output_table[optimizer_run_name][dataset_version][dataset_name]['best_model_ids']:

                with open(paths[optimizer_run_name]['models'].format(model_id, 'json')) as file:
                    json_string = file.read()

                model = json.loads(json_string)

                trained_model = load(paths[optimizer_run_name]['models'].format(model_id, 'jolib'))
                with open(paths[optimizer_run_name]['models'].format(model_id, 'json'), 'r', encoding='utf-8') as file:
                    model_meta_dump = file.read()
                model_meta = json.loads(model_meta_dump)

                inputs = model_meta['inputs']

                # Pearson on valid. test
                features = []
                for method in inputs:
                    index = configuration_index(method['args'], [x['args'] for x in
                                                                 model_meta['validation']['features'][
                                                                     method['method_name']]])
                    if index == -1:
                        raise ValueError('Args not found')
                    features.append(
                        model_meta['validation']['features'][method['method_name']][index]['values']
                    )
                features = transpose(features)

                labels = model_meta['validation']['labels']

                pearson_value = pearson(trained_model.predict(features), labels)
                model['validation_pearson'] = pearson_value

                models.append(model)

            best_model = reduce(lambda a, b: a if a['validation_pearson'] > b['validation_pearson'] else b, models)
            output_table[optimizer_run_name][dataset_version][dataset_name]['best_model'] = best_model

for optimizer_run_name in output_table:
    print(optimizer_run_name)
    for dataset_version in output_table[optimizer_run_name]:
        print("\t" + dataset_version)
        for dataset_name in output_table[optimizer_run_name][dataset_version]:
            print("\t\t" + dataset_name)
            print("\t\t\t" + str(output_table[optimizer_run_name][dataset_version][dataset_name]['best_model']['validation_pearson']))
            print("\t\t\t" + str(output_table[optimizer_run_name][dataset_version][dataset_name]['best_model']['type']))
            print("\t\t\t" + str(output_table[optimizer_run_name][dataset_version][dataset_name]['best_model']['hyperparams']))
            print("\t\t\t" + str(output_table[optimizer_run_name][dataset_version][dataset_name]['best_model']['inputs']))
            exit()


for optimizer_run_name in output_table:
    overleaf_format = ''
    for dataset_name in output_table[optimizer_run_name]['raw']:
        overleaf_format += '\\hline\n'
        overleaf_format += dataset_name + ' & \\multicolumn{2}{|c|}{Pearson} \\\\\n'
        overleaf_format += '\\cline{2-3}\n'
        overleaf_format += ' & ' + trim_number(output_table[optimizer_run_name]['raw'][dataset_name]['best_model']['validation_pearson']) + ' & ' + trim_number(output_table[optimizer_run_name]['lemma'][dataset_name + '_lemma']['best_model']['validation_pearson']) + ' \\\\\n'
        overleaf_format += '\\cline{2-3}\n'
        overleaf_format += ' & \\multicolumn{2}{|c|}{Model type} \\\\\n'
        overleaf_format += '\\cline{2-3}\n'
        overleaf_format += ' & ' + abbreviate_model_type(output_table[optimizer_run_name]['raw'][dataset_name]['best_model']['type']) + ' & ' + abbreviate_model_type(output_table[optimizer_run_name]['lemma'][dataset_name + '_lemma']['best_model']['type']) + ' \\\\\n'
        overleaf_format += '\\cline{2-3}\n'
        overleaf_format += ' & \\multicolumn{2}{|c|}{Hyperparameters} \\\\\n'
        overleaf_format += '\\cline{2-3}\n'


        # Hyperparams
        hyperparams_raw_dict = output_table[optimizer_run_name]['raw'][dataset_name]['best_model']['hyperparams']
        hyperparams_raw = ['\\textit{' + key + ':' + str(hyperparams_raw_dict[key]) + '}' for key in list(hyperparams_raw_dict.keys()) if key != 'n_jobs']

        hyperparams_lemma_dict = output_table[optimizer_run_name]['lemma'][dataset_name + '_lemma']['best_model']['hyperparams']
        hyperparams_lemma = ['\\textit{' + key + ':' + str(hyperparams_lemma_dict[key]) + '}' for key in list(hyperparams_lemma_dict.keys()) if key != 'n_jobs']

        for i in range(min(len(hyperparams_raw), len(hyperparams_lemma))):
            overleaf_format += ' & ' + hyperparams_raw[i] + ' & ' + hyperparams_lemma[i] + ' \\\\\n'
        if len(hyperparams_raw) > len(hyperparams_lemma):
            for item in hyperparams_raw[len(hyperparams_lemma):]:
                overleaf_format += ' & ' + item + ' & \\\\\n'
        elif len(hyperparams_raw) < len(hyperparams_lemma):
            for item in hyperparams_lemma[len(hyperparams_raw):]:
                overleaf_format += ' & ' + item + ' & \\\\\n'

        overleaf_format += '\\cline{2-3}\n'
        overleaf_format += ' & \\multicolumn{2}{|c|}{Features} \\\\\n'
        overleaf_format += '\\cline{2-3}\n'

        # Features
        features_raw = ['\\textit{' + x['method_name'] + '}' for x in output_table[optimizer_run_name]['raw'][dataset_name]['best_model']['inputs']]
        features_lemma = ['\\textit{' + x['method_name'] + '}' for x in output_table[optimizer_run_name]['lemma'][dataset_name + '_lemma']['best_model']['inputs']]
        for i in range(min(len(features_raw), len(features_lemma))):
            overleaf_format += ' & ' + features_raw[i] + ' & ' + features_lemma[i] + ' \\\\\n'
        if len(features_raw) > len(features_lemma):
            for item in features_raw[len(features_lemma):]:
                overleaf_format += ' & ' + item + ' & \\\\\n'
        elif len(features_raw) < len(features_lemma):
            for item in features_lemma[len(features_raw):]:
                overleaf_format += ' & ' + item + ' & \\\\\n'

    print('-' * 50)
    print(overleaf_format.replace('_', '\\_'))


