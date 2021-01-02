# Runnable script calculating values for each dataset for each method and persisting them
# Already persisted values are not calculated again nor persisted

import re
from os import listdir, getcwd
from os.path import isfile, join

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from model_management.sts_method_pool import sts_method_pool
from model_management.sts_method_value_persistor import input_folder, predict_and_persist_values, persist_gold_standard

# Let's prepare regexes to be used throughout this script.
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?\.txt")

# Let's prepare list of dataset files for which values should be calculated and persisted
input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]

# Loop over each dataset to calculate and persist values for all methods
for dataset_name in input_dataset_files:
    # Persist gold standard first - no need to calculate anything
    persist_gold_standard(dataset_name)

    # Loop over each method we know
    for sts_method_name in sts_method_pool:
        # Predict and persist values of this dataset of this method if needed
        predict_and_persist_values(sts_method_pool[sts_method_name], dataset_name)
