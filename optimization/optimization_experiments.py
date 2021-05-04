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
from basic_similarity_methods.vector_based_create_vectors import average
from complex_similarity_methods.dataset_fragmentation import FragmentedDatasetCV, FragmentedDatasetSuper
from complex_similarity_methods.dataset_split_ratio import DatasetSplitRatio

from complex_similarity_methods.regression_methods_core import prepare_training_data, train_n_test
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from dataset_modification_scripts.dataset_pool import dataset_pool
from evaluation.evaluate_regression_metrics import pearson
from model_management.sts_model_pool import model_types

# Config
dataset_split_ratio = DatasetSplitRatio(0.70, 0.30)
cross_validation_fold_count = 4
fitness_metric = {
    'name': 'pearson',
    'method': pearson
}
weights = {
    'elements': ['test', 'validation'],
    'values': [1, 3]
}

# Storage
algorithm_run = {
    'run_id': 1,
    'main': {}
}

# Dataset - specific
split_dataset_master = None
sorted_method_group_names = None
method_count = None
method_param_counts = None

# Dataset&Model - specific
sorted_arg_names = None
arg_possibility_counts = None
best_model = None


# Beehive looks for minimum. Make this so that lowest value of this function means the best solution
def solution_evaluator(vector):
    temp_vector = list(map(lambda x: int(x), vector))
    param_dict = {}

    # Hyperparameters of model
    for i in range(len(sorted_arg_names)):
        param_dict[sorted_arg_names[i]] = model_type['args'][sorted_arg_names[i]][temp_vector[i]]

    # Features to be fed to model
    inputs = []
    temp_vector_starting_index = len(sorted_arg_names)
    for i in range(method_count):
        if temp_vector[temp_vector_starting_index + i] < method_param_counts[i]:
            inputs.append({
                'method_name': sorted_method_group_names[i],
                'values': split_dataset_master.Train.features[sorted_method_group_names[i]][temp_vector[temp_vector_starting_index + i]]['values']
            })

    if len(inputs) < 2:
        return 2

    # Data for model
    dataset_fragments = FragmentedDatasetCV(inputs, split_dataset_master.Train.labels, cross_validation_fold_count)
    metric_values_test = []
    for k in range(len(dataset_fragments.folds)):
        model_data = dataset_fragments.produce_split_dataset(k).produce_sklearn_ready_data()

        # ML Model
        model = model_type['model'](**param_dict)
        model.fit(model_data.train.features, model_data.train.labels)
        metric_values_test.append(fitness_metric['method'](model_data.test.labels, model.predict(model_data.test.features)))

    metric_avg_test = average(metric_values_test)

    # print(metric_values_test)
    # print(metric_values_valid)
    # print(metric_avg_test)
    # print(metric_avg_valid)
    # print(metric_total)

    # print('Validation.Labels: {}'.format(len(dataset_fragments.validation_data.labels)))
    # print('Validation.Features: {}x{}'.format(len(dataset_fragments.validation_data.features), len(dataset_fragments.validation_data.features[0])))
    # for i in range(len(dataset_fragments.folds)):
    #     print('Fold {}'.format(i + 1))
    #     print('\tLabels: {}'.format(len(dataset_fragments.folds[i].labels)))
    #     print('\tFeatures: {}x{}'.format(
    #         len(dataset_fragments.folds[i].features),
    #         len(dataset_fragments.folds[i].features[0]))
    #     )

    # The algorithm minimizes the fitness value -> adjust the metric for this
    fitness = 1 - metric_avg_test if metric_avg_test is not None else 2

    global best_model
    if best_model is None or best_model['fitness'] > fitness:
        best_model = {
            'vector': vector,
            'fitness': fitness
        }

    return fitness


# ---- SOLVE TEST CASE WITH ARTIFICIAL BEE COLONY ALGORITHM

def run_optimization():
    # creates optimizer model
    model = Hive.BeeHive(
                         lower=[0] * (len(arg_possibility_counts) + method_count),
                         upper=list(map(lambda x: x - 1, arg_possibility_counts)) +
                               list(map(lambda x: 2 * x - 1, method_param_counts)),
                         fun=solution_evaluator,
                         numb_bees=5,
                         max_itrs=10
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
    print("Best pearson by ABC: {}\nAchieved by {}\n Best History: {}\n Mean History: {}".format(1-model.best, model.solution, cost['best'], cost['mean']))

    x = model.solution
    #print('Solution is: {}'.format(x))
    print("-" * 40 + "\nBest model - person: {}\nSolution: {}".format(1 - best_model['fitness'], best_model['vector']))
    exit()


dataset_counter = 1
dataset_counter_max = len(list(dataset_pool.keys())) * len(dataset_pool[list(dataset_pool.keys())[0]])

model_in_dataset_counter_max = len(model_types)

total_counter = 1
total_counter_max = dataset_counter_max * model_in_dataset_counter_max

for key in dataset_pool:
    algorithm_run['main'][key] = {}
    for dataset in dataset_pool[key]:
        algorithm_run['main'][key][dataset.name] = {}

        persisted_methods_temp = dataset.load_values()
        gold_values_temp = [round(x/5, ndigits=3) for x in persisted_methods_temp[gold_standard_name][0]['values']]
        del persisted_methods_temp[gold_standard_name]

        split_dataset_master = FragmentedDatasetSuper(persisted_methods_temp, gold_values_temp, dataset_split_ratio)
        sorted_method_group_names = sorted(persisted_methods_temp.keys())
        method_count = len(sorted_method_group_names)

        method_param_counts = [len(persisted_methods_temp[sorted_method_group_names[i]]) for i in range(method_count)]

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

            best_model = None
            run_optimization()

            model_in_dataset_counter = model_in_dataset_counter + 1
            total_counter = total_counter + 1

        dataset_counter = dataset_counter + 1
