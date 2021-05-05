# Runnable script calculating values for each dataset for each method and persisting them
# Already persisted values are not calculated again nor persisted

from os import getcwd

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from dataset_modification_scripts.dataset_pool import dataset_pool

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
methods_to_delete = ['mlipns']
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods
    for dataset in dataset_pool[key]:
        values = dataset.load_values()
        deleted = False

        for method_name in methods_to_delete:
            if method_name in values:
                print('Found {} in {}. TIME TO DELETE!'.format(method_name, dataset.name))
                del values[method_name]
                deleted = True

        if deleted:
            dataset.persist_values(values)
