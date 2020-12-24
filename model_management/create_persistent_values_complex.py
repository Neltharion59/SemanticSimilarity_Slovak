import re
from os import listdir, getcwd
from os.path import isfile, join

import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
from complex_similarity_methods.regression_methods_core import prepare_training_data

conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from model_management.sts_model_pool import sts_model_pool
from model_management.sts_method_value_persistor import input_folder, predict_and_persist_values, persist_gold_standard


for dataset in sts_model_pool:
    print("Dataset {}".format(dataset))

    for model_name in sts_model_pool[dataset]:
        print("Dataset {}, model {}".format(dataset, sts_model_pool[dataset][model_name].name))

        # TODO if model or its (partial) values are already persisted

        # Else predict and persist values and model
        print("Model not persisted - need to do that now")
        if not sts_model_pool[dataset][model_name].trained:
            print("About to train the model as it is not trained yet")
            x_train, x_test, y_train, y_test = prepare_training_data(
                dataset,
                sts_model_pool[dataset][model_name].input_names
            )
            sts_model_pool[dataset][model_name].train(x_train, x_test, y_train, y_test)
            print("Model trained")
        predict_and_persist_values(sts_model_pool[dataset][model_name], dataset)
        # TODO persist model
