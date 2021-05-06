# Runnable script calculating values for each dataset for each method and persisting them
# Already persisted values are not calculated again nor persisted

from os import getcwd

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from model_management.sts_method_pool import sts_method_pool
from dataset_modification_scripts.dataset_pool import dataset_pool, find_dataset_by_name


def dict_match(dict1, dict2):

    for key in dict1:
        if key not in dict2:
            return False
        if dict1[key] != dict2[key]:
            return False

    for key in dict2:
        if key not in dict1:
            return False
        if dict1[key] != dict2[key]:
            return False

    return True


for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods
    for dataset in dataset_pool[key]:
        print('Processing dataset {}'.format(dataset.name))

        # Persist gold standard first - no need to calculate anything
        dataset.persist_gold_standard()

        # THIS ASSUMES THAT ALL ARG COMBINATIONS ARE ORDERED THE SAME FOR ALL DATASETS AND ALL VALUES ARE CALCULATED 4 ALL DATASETS
        if dataset.name in ['semeval-all', 'semeval-all_lemma', 'all', 'all_lemma']:
            sub_datasets = []

            if dataset.name == 'semeval-all':
                sub_datasets = ['semeval-2012', 'semeval-2013', 'semeval-2014', 'semeval-2015', 'semeval-2016']
            elif dataset.name == 'semeval-all_lemma':
                sub_datasets = ['semeval-2012_lemma', 'semeval-2013_lemma', 'semeval-2014_lemma', 'semeval-2015_lemma',
                                'semeval-2016_lemma']
            elif dataset.name == 'all':
                sub_datasets = ['semeval-2012', 'semeval-2013', 'semeval-2014', 'semeval-2015', 'semeval-2016', 'sick']
            elif dataset.name == 'all_lemma':
                sub_datasets = ['semeval-2012_lemma', 'semeval-2013_lemma', 'semeval-2014_lemma', 'semeval-2015_lemma',
                                'semeval-2016_lemma', 'sick_lemma']
            else:
                raise ValueError('No sub datasets for composite dataset {}'.format(dataset.name))

            first_dataset = find_dataset_by_name(key, sub_datasets[0])
            if first_dataset is None:
                raise ValueError('Sub dataset not found: {} in {}'.format(sub_datasets[0], dataset.name))
            merged_dict = first_dataset.load_values()
            if len(list(merged_dict.keys())):
                raise ValueError('Sub dataset empty: {} in {}'.format(sub_datasets[0], dataset.name))

            for sub_dataset_name in sub_datasets[1:]:

                sub_dataset = find_dataset_by_name(key, sub_dataset_name)
                if sub_dataset is None:
                    raise ValueError('Sub dataset not found: {} in {}'.format(sub_dataset_name, dataset.name))
                current_dict = sub_dataset.load_values()
                if len(list(current_dict.keys())) == 0:
                    raise ValueError('Sub dataset empty: {} in {}'.format(sub_dataset_name, dataset.name))

                for method_name in current_dict:
                    if method_name not in merged_dict:
                        print('For dataset {} skipping {}'.format(dataset.name, method_name))
                        continue

                    for i in range(len(current_dict[method_name])):
                        for j in range(len(merged_dict[method_name])):
                            if dict_match(current_dict[method_name][i]['args'], merged_dict[method_name][i]['args']):
                                merged_dict[method_name][j]['values'] = merged_dict[method_name][j]['values'] + current_dict[method_name][i]['values']

                for method_name in merged_dict:
                    if method_name not in current_dict:
                        print('For dataset {} removing method {}'.format(dataset.name, method_name))
                        del method_name[method_name]

            max_len = 0
            for method_name in merged_dict:
                for record in merged_dict[method_name]:
                    max_len = max(max_len, len(record['values']))

            for method_name in merged_dict:
                for i in reversed(range(len(merged_dict[method_name]))):
                    if len(merged_dict[method_name][i]['values']) < max_len:
                        print('Removing configuration of method {} for dataset {}. Configuration: {}'.format(method_name, dataset.name,merged_dict[method_name][i]['args']))
                        del merged_dict[method_name][i]

            for method_name in merged_dict:
                if len(merged_dict[method_name]) == 0:
                    print('Removing entire method {} for dataset {}.'.format(method_name, dataset.name))
                    del merged_dict[method_name]

            dataset.persist_values(merged_dict)

        # Loop over each method we know
        for sts_method_name in sts_method_pool:
            for sts_method in sts_method_pool[sts_method_name]:

                if 'corpus' in sts_method.args:
                    if 'lemma' in dataset.name:
                        sts_method.args['corpus'] = sts_method.args['corpus'].replace('_sk.txt', '_sk_lemma.txt')
                    else:
                        sts_method.args['corpus'] = sts_method.args['corpus'].replace('_sk_lemma.txt', '_sk.txt')
                else:
                    dataset.predict_and_persist_values(sts_method)
