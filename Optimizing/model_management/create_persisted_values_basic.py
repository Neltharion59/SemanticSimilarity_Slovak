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


# Function to check if two dicts contain the same keys and values.
# Params: dict<str, any>, dict<str, any>
# Return: bool
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


# For each dataset version (raw vs. lemma)
for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods
    for dataset in dataset_pool[key]:

        print('Processing dataset {}'.format(dataset.name))

        # Persist gold standard first - no need to calculate anything
        dataset.persist_gold_standard()

        # Some datasets are just combinations of other datasets. Instead of calculating again,
        # let's just combine the already calculated values.
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

            merged_dict = dataset.load_values()
            # Combine the sub datasets.
            for sub_dataset_name in sub_datasets:

                sub_dataset = find_dataset_by_name(key, sub_dataset_name)
                if sub_dataset is None:
                    raise ValueError('Sub dataset not found: {} in {}'.format(sub_dataset_name, dataset.name))
                current_dict = sub_dataset.load_values()
                if len(list(current_dict.keys())) == 0:
                    raise ValueError('Sub dataset empty: {} in {}'.format(sub_dataset_name, dataset.name))

                for method_name in current_dict:
                    if method_name == 'gold_standard':
                        continue

                    if method_name not in merged_dict:
                        merged_dict[method_name] = current_dict[method_name]
                        continue

                    for i in range(len(current_dict[method_name])):
                        for j in range(len(merged_dict[method_name])):
                            if dict_match(current_dict[method_name][i]['args'], merged_dict[method_name][j]['args']):
                                merged_dict[method_name][j]['values'] = merged_dict[method_name][j]['values'] + current_dict[method_name][i]['values']
                                break
            # If some of combined values have incosistent length, let's find out and remove them.
            # -----------------------------------------------------------------------------------
            # First, see the maximum amount of records any method has.
            max_len = 0
            for method_name in merged_dict:
                for record in merged_dict[method_name]:
                    max_len = max(max_len, len(record['values']))
            # If any configuration is inconsistent length, let's get rid of it.
            print('Desired length is {}.'.format(max_len))
            for method_name in merged_dict:
                for i in reversed(range(len(merged_dict[method_name]))):
                    if len(merged_dict[method_name][i]['values']) < max_len:
                        print('Length: {} | Removing configuration of method {} for dataset {}. Configuration: {}'.format(len(merged_dict[method_name][i]['values']), method_name, dataset.name,merged_dict[method_name][i]['args']))
                        merged_dict[method_name].pop(i)
            # If removing configurations lead to having empty emthods, let's get rid of them too.
            for method_name in merged_dict:
                if len(merged_dict[method_name]) == 0:
                    print('Removing entire method {} for dataset {}.'.format(method_name, dataset.name))
                    merged_dict.pop(method_name)
            # Persist values to disk.
            dataset.persist_values(merged_dict)
        # We are not in dataset composed of other datasets, so let's normally compute the values.
        else:
            # Loop over each method we know.
            for sts_method_name in sts_method_pool:
                for sts_method in sts_method_pool[sts_method_name]:

                    # If we are dealing with lematized dataset, let's adjust corpora name in args,
                    # if the method is vector-based
                    if 'corpus' in sts_method.args:
                        if 'lemma' in dataset.name:
                            sts_method.args['corpus'] = sts_method.args['corpus'].replace('_sk.txt', '_sk_lemma.txt')
                        else:
                            sts_method.args['corpus'] = sts_method.args['corpus'].replace('_sk_lemma.txt', '_sk.txt')
                    else:
                        dataset.predict_and_persist_values(sts_method)
