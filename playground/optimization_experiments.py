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


optionsDCT = {
    'splitter': ['random', 'best'],
    'max_depth': list(range(5, 20)) + [None],
    'min_samples_split': list(range(2, 50)),
    'min_samples_leaf': list(range(1, 30)),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_leaf_nodes': list(range(2, 10))
}
dataset = 'dataset_sick_all_sk.txt'
persisted_methods = get_persisted_method_types(dataset)
persisted_methods = list(filter(lambda x: '___' in x, persisted_methods))
grouped_methods = group_method_names(persisted_methods)
sorted_method_group_names = sorted(grouped_methods.keys())
method_count = len(sorted_method_group_names)
method_param_counts = [len(grouped_methods[sorted_method_group_names[i]]) for i in range(method_count)]

sorted_arg_names = sorted(optionsDCT.keys())

arg_possibility_counts = list(map(lambda x: len(optionsDCT[x]), sorted_arg_names))


# Beehive looks for minimum. Make this so that lowest value of this function means the best solution
def evaluator(vector):

    temp_vector = list(map(lambda x: int(round(x, 0)), vector))
    #print(len(temp_vector))
    param_dict = {}
    for i in range(len(sorted_arg_names)):
        param_dict[sorted_arg_names[i]] = optionsDCT[sorted_arg_names[i]][temp_vector[i]]
        #print(i)
    model = DecisionTreeRegressor(**param_dict)

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

def run():

    # creates model
    model = Hive.BeeHive(
                         lower=[0] * (len(arg_possibility_counts) + method_count),
                         upper=list(map(lambda x: x - 1, arg_possibility_counts)) +
                               list(map(lambda x: 2 * x - 1, method_param_counts)),
                         fun=evaluator,
                         numb_bees=10,
                         max_itrs=5
                        )

    # runs model
    cost = model.run()

    # plots convergence
    Utilities.ConvergencePlot(cost)

    # prints out best solution
    print("Fitness Value ABC: {0}".format(model.best))
    print("Best pearson by ABC: {0}".format(1-model.best))

    x = model.solution
    print(x)


if __name__ == "__main__":
    run()
