# Runnable script calculating values for each dataset for each method and persisting them
# Already persisted values are not calculated again nor persisted
import math
from os import getcwd

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from dataset_modification_scripts.dataset_pool import dataset_pool


for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods
    for dataset in dataset_pool[key]:
        values = dataset.load_values()

        for method_name in values:
            for i in range(len(values[method_name])):
                values[method_name][i]['values'] = [x if not math.isnan(x) else 0 for x in values[method_name][i]['values']]

        dataset.persist_values(values)
