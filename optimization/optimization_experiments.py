import re
from os import listdir, getcwd
from os.path import isfile, join

import sys

import math
import numpy as np
from functools import reduce
from operator import add

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from Hive import Utilities
from Hive import Hive
from complex_similarity_methods.dataset_fragmentation import FragmentedDatasetCV
from complex_similarity_methods.dataset_split_ratio import DatasetSplitRatio

from complex_similarity_methods.regression_methods_core import prepare_training_data, train_n_test
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from dataset_modification_scripts.dataset_pool import dataset_pool
from model_management.sts_model_pool import model_types

# Config
dataset_split_ratio = DatasetSplitRatio(0.70, 0.30)

# Dataset - specific
persisted_methods = None
sorted_method_group_names = None
method_count = None
method_param_counts = None
gold_values = None
# Dataset&Model - specific
sorted_arg_names = None
arg_possibility_counts = None


# Beehive looks for minimum. Make this so that lowest value of this function means the best solution
def solution_evaluator(vector):
    temp_vector = list(map(lambda x: int(x), vector))
    param_dict = {}
    for i in range(len(sorted_arg_names)):
        param_dict[sorted_arg_names[i]] = model_type['args'][sorted_arg_names[i]][temp_vector[i]]

    model = model_type['model'](**param_dict)

    inputs = []
    temp_vector_starting_index = len(sorted_arg_names)
    for i in range(method_count):
        if temp_vector[temp_vector_starting_index + i] < method_param_counts[i]:
            inputs.append({
                'method_name': sorted_method_group_names[i],
                'values': persisted_methods[sorted_method_group_names[i]][temp_vector[temp_vector_starting_index + i]]['values']
            })

    print('ja')
    dataset_fragments = FragmentedDatasetCV(inputs, gold_values, dataset_split_ratio, 4)
    print('aj')
    exit()

    x_train, x_test, y_train, y_test = prepare_training_data(
                dataset,
                inputs,
                print_head=False
            )

    #print('Hive::solution_evaluator - input names : {}'.format(input_names))
    #print('Hive::solution_evaluator: - param dict: {}'.format(param_dict))

    if len(input_names) < 2:
        return 2

    metrics = train_n_test(x_train, x_test, y_train, y_test, model, verbose=False)
    #print("Model with pearson: ", metrics['PEARSON'][0])

    return 1 - metrics['PEARSON'][0] if metrics['PEARSON'][0] is not None else 2


# ---- SOLVE TEST CASE WITH ARTIFICIAL BEE COLONY ALGORITHM

def run_optimization():

    # creates model
    model = Hive.BeeHive(
                         lower=[0] * (len(arg_possibility_counts) + method_count),
                         upper=list(map(lambda x: x - 1, arg_possibility_counts)) +
                               list(map(lambda x: 2 * x - 1, method_param_counts)),
                         fun=solution_evaluator,
                         numb_bees=2,
                         max_itrs=2
                        )

    # runs model
    cost = model.run()

    # plots convergence
    #print(cost)
    # Make COST contain Pearson instead of fitness function - graph will show Pearson nicely
    for x in cost:
        cost[x] = list(map(lambda i: 1-i, cost[x]))
    #Utilities.ConvergencePlot(cost)

    # prints out best solution
    #print("Fitness Value ABC: {0}".format(model.best))
    print("Best pearson by ABC: {0}".format(1-model.best))

    x = model.solution
    #print('Solution is: {}'.format(x))


dataset_counter = 1
dataset_counter_max = len(list(dataset_pool.keys())) * len(dataset_pool[list(dataset_pool.keys())[0]])

model_in_dataset_counter_max = len(model_types)

total_counter = 1
total_counter_max = dataset_counter_max * model_in_dataset_counter_max

for key in dataset_pool:
    for dataset in dataset_pool[key]:

        persisted_methods = dataset.load_values()
        gold_values = [round(x/5, ndigits=3) for x in persisted_methods[gold_standard_name][0]['values']]
        del persisted_methods[gold_standard_name]

        sorted_method_group_names = sorted(persisted_methods.keys())
        method_count = len(sorted_method_group_names)

        method_param_counts = [len(persisted_methods[sorted_method_group_names[i]]) for i in range(method_count)]

        model_in_dataset_counter = 1
        for model_type in model_types:

            sorted_arg_names = sorted(model_type['args'].keys())
            arg_possibility_counts = list(map(lambda x: len(model_type['args'][x]), sorted_arg_names))

            print('TRAINING MODEL {}/{} | DATASET {}: {}/{} | MODEL {}: {}/{} | TOTAL {}%'.format(
                total_counter, total_counter_max,
                dataset.name, dataset_counter, dataset_counter_max,
                model_type['name'], model_in_dataset_counter, model_in_dataset_counter_max,
                round(total_counter / total_counter_max, ndigits=4) * 100
            ))

            run_optimization()

            model_in_dataset_counter = model_in_dataset_counter + 1
            total_counter = total_counter + 1

        dataset_counter = dataset_counter + 1
