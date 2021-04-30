# Runnable script calculating values for each dataset for each method and persisting them
# Already persisted values are not calculated again nor persisted

import re
from os import getcwd

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from model_management.sts_method_pool import sts_method_pool
from dataset_modification_scripts.dataset_pool import dataset_pool

for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods
    for dataset in dataset_pool[key]:
        # Persist gold standard first - no need to calculate anything
        dataset.persist_gold_standard()
        # Loop over each method we know
        for sts_method_name in sts_method_pool:
            for sts_method in sts_method_pool[sts_method_name]:
                dataset.predict_and_persist_values(sts_method)
