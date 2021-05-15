# Runnable script deleting values of specifieds methods and configurations.
from os import getcwd

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from dataset_modification_scripts.dataset_pool import dataset_pool

# ---------------------------------------------------------------------------------------
# ----------------- SPECIFY METHODS AND CONFIGS TO BE DELETED ---------------------------
# ---------------------------------------------------------------------------------------
methods_to_delete = ['mlipns']
confs_to_delete = [
    {
        'method_name': 'minkowski',
        'args': {
            'p': 3.141592653589793
        }
    }
]
confs_only_to_delete = {
    'corpus': [
        '2016_web_sk.txt', '2016_wikipedia_sk.txt', '2018_wiki_sk.txt', '2020_news_sk.txt',
        '2016_web_sk_lemma.txt', '2016_wikipedia_sk_lemma.txt', '2018_wiki_sk_lemma.txt', '2020_news_sk_lemma.txt'
    ]
}
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# Check if second dict is a subset of first dict.
# Params: dict<str, any>, dict<str, any>
# Return: bool
def dict_subset(super_dict, sub_dict):

    for key in sub_dict:
        if key not in super_dict:
            return False
        if sub_dict[key] != super_dict[key]:
            return False

    return True


# For each dataset version.
for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods.
    for dataset in dataset_pool[key]:
        # Load persisted values for given dataset.
        values = dataset.load_values()
        deleted = False
        # Check methods to be deleted.
        for method_name in methods_to_delete:
            if method_name in values:
                print('Found {} in {}. TIME TO DELETE!'.format(method_name, dataset.name))
                values.pop(method_name)
                deleted = True
        # Check configurations to be deleted (single value for each param).
        for conf_to_delete in confs_to_delete:
            if conf_to_delete['method_name'] in values:
                for i in reversed(range(len(values[conf_to_delete['method_name']]))):
                    if dict_subset(values[conf_to_delete['method_name']][i]['args'], conf_to_delete['args']):
                        print('Deleting configuration of method {} for dataset {}. Configuration: {}'.format(conf_to_delete['method_name'] , dataset.name, values[conf_to_delete['method_name']][i]['args']))
                        values[conf_to_delete['method_name']].pop(i)
                        deleted = True
        # Check configurations to be deleted (multiple values for each param).
        for method_name in values:
            for i in reversed(range(len(values[method_name]))):
                for arg_name in confs_only_to_delete:
                    if arg_name in values[method_name][i]['args'] and values[method_name][i]['args'][arg_name] in confs_only_to_delete[arg_name]:
                        print('Deleting configuration of method {} for dataset {}. Configuration: {}'.format(method_name, dataset.name, values[method_name][i]['args']))
                        values[method_name].pop(i)
                        deleted = True
        # Finally, persist the changes.
        if deleted:
            dataset.persist_values(values)
