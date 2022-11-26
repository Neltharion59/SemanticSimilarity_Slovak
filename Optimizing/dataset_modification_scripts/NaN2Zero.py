# Runnable script rewriting NaN values as 0 (due to some errors in computation, this was necessary).
import math
from os import getcwd

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from dataset_modification_scripts.dataset_pool import dataset_pool

# For each dataset version (raw vs. lemma)
for key in dataset_pool:
    # Loop over each dataset to calculate and persist values for all methods
    for dataset in dataset_pool[key]:
        values = dataset.load_values()
        # Check each method
        for method_name in values:
            # Check each configuration
            for i in range(len(values[method_name])):
                values[method_name][i]['values'] = [x if not math.isnan(x) else 0 for x in values[method_name][i]['values']]
        # Finally persist the corrected  values
        dataset.persist_values(values)
