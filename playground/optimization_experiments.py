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

from complex_similarity_methods.regression_methods_core import prepare_training_data, train_n_test
from model_management.sts_method_value_persistor import get_persisted_method_types
from dataset_modification_scripts.dataset_pool import dataset_pool
from model_management.sts_model_pool import model_types

#' Dataset - specific
persisted_methods = None
persisted_methods = None
grouped_methods = None
sorted_method_group_names = None
method_count = None
method_param_counts = None
#'Dataset&Model - specific
sorted_arg_names = None
arg_possibility_counts = None

def group_method_names(method_names):
    output_dict = {}
    for method_name in method_names:
        method = method_name.split('___')[0]
        params = method_name.split('___')[1]
        if method not in output_dict:
            output_dict[method] = []
        if params not in output_dict[method]:
            output_dict[method].append(params)
    return output_dict


# Beehive looks for minimum. Make this so that lowest value of this function means the best solution
def solution_evaluator(vector):

    temp_vector = list(map(lambda x: int(round(x, 0)), vector))
    #print(len(temp_vector))
    param_dict = {}
    for i in range(len(sorted_arg_names)):
        param_dict[sorted_arg_names[i]] = model_type['args'][sorted_arg_names[i]][temp_vector[i]]
        #print(i)
    print(model_type['model'])
    print(param_dict)
    model = model_type['model'](**param_dict)

    input_names = []
    for i in range(method_count):
        #print('i: ', i)
        if temp_vector[len(sorted_arg_names) + i] < method_param_counts[i]:
            #print('index to temp vector: ', len(sorted_arg_names) + i)
            #print('index to param array of group: ', temp_vector[len(sorted_arg_names) + i])
            #print('given group has # arg options: ', method_param_counts[i])
            input_names.append(sorted_method_group_names[i] +
                               '___' +
                               grouped_methods[sorted_method_group_names[i]]
                               [temp_vector[len(sorted_arg_names) + i]])
    x_train, x_test, y_train, y_test = prepare_training_data\
            (
                dataset,
                input_names,
                print_head=False
            )
    print("----------------------------------------------------------------------------------")
    print(input_names)
    print(param_dict)

    if len(input_names) < 2:
        return 2

    metrics = train_n_test(x_train, x_test, y_train, y_test, model)
    print("Model with pearson: ", metrics['PEARSON'][0])

    return 1 - metrics['PEARSON'][0]


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

    # # runs model
    # cost = model.run()
    #
    # # plots convergence
    # print("COST START")
    # print(cost)
    # # Make COST contain Pearson instead of fitness function - graph will show Pearson nicely
    # for x in cost:
    #     cost[x] = list(map(lambda i: 1-i, cost[x]))
    # print("COST END")
    # Utilities.ConvergencePlot(cost)
    #
    # # prints out best solution
    # print("Fitness Value ABC: {0}".format(model.best))
    # print("Best pearson by ABC: {0}".format(1-model.best))
    #
    # x = model.solution
    # print(x)


for dataset in dataset_pool:

    persisted_methods = get_persisted_method_types(dataset)
    persisted_methods = list(filter(lambda x: '___' in x, persisted_methods))
    grouped_methods = group_method_names(persisted_methods)
    sorted_method_group_names = sorted(grouped_methods.keys())
    method_count = len(sorted_method_group_names)
    method_param_counts = [len(grouped_methods[sorted_method_group_names[i]]) for i in range(method_count)]

    for model_type in model_types:
        print((dataset.name,
               model_type['name']))

        sorted_arg_names = sorted(model_type['args'].keys())
        print(sorted_arg_names)

        arg_possibility_counts = list(map(lambda x: len(model_type['args'][x]), sorted_arg_names))
        print(arg_possibility_counts)
        print('Posibilities: {}'.format(reduce(lambda x, y: x * y, arg_possibility_counts)))

        run_optimization()

