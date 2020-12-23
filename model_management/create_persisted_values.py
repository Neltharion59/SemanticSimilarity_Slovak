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

dataset_input_file_name_pattern = re.compile(".*_sk\.txt")

input_dataset_files = [x for x in listdir(input_folder) if isfile(join(input_folder, x)) and dataset_input_file_name_pattern.match(x)]

for dataset_name in input_dataset_files:
    persist_gold_standard(dataset_name)

    for sts_method_name in sts_method_pool:
        predict_and_persist_values(sts_method_pool[sts_method_name], dataset_name)
