import math
import json
import os
import sys
from datetime import datetime

from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from Hive import Hive
from util.math import average
from complex_similarity_methods.dataset_fragmentation import FragmentedDatasetCV, FragmentedDatasetSuper
from complex_similarity_methods.dataset_split_ratio import DatasetSplitRatio

from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from dataset_modification_scripts.dataset_pool import dataset_pool
from evaluation.evaluate_regression_metrics import pearson
from model_management.persistent_id_generator import PersistentIdGenerator
from model_management.sts_model_pool import model_types

from playsound import playsound

path_to_optimizer_run_record_folder = './../resources/optimizer_runs/'

# Config
dataset_split_ratio = DatasetSplitRatio(0.70, 0.30)
cross_validation_fold_count = 4
fitness_metric = {
    'name': 'pearson',
    'method': pearson
}
bee_count = 80
iteration_cap = 500

# Storage
algorithm_run = {
    'run_id': PersistentIdGenerator('optimizer_run').next_id(),
    'config': {
        'train_ratio': dataset_split_ratio.train_ratio,
        'valid_ratio': dataset_split_ratio.validation_ratio,
        'CV_fold_count': cross_validation_fold_count,
        'fitness_metric': fitness_metric['name'],
        'optimizer': {
            'name': 'ABC',
            'config': {
                'bee_count': bee_count,
                'iteration_cap': iteration_cap
            }
        }
    },
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
model_id_generator = PersistentIdGenerator('model_persistence')


# Persistence - related util function
def load_optimizer_run(run_id):
    try:
        with open(
            '{}optimizer_run_{}.txt'.format(path_to_optimizer_run_record_folder, run_id),
            'r',
            encoding='utf-8'
        ) as file:
            txt_dump = file.read()
        print('--- OLD --- Run with id {} exists. Let\'s continue where we dropped off'.format(run_id))
        global algorithm_run
        algorithm_run = json.loads(txt_dump)
    except FileNotFoundError:
        print('--- NEW --- Run with id {} does not exist. Let\'s start a new one'.format(run_id))


# Persistence - related util function
def persist_optimizer_run():
    txt_dump = json.dumps(algorithm_run)
    with open(
        '{}optimizer_run_{}.txt'.format(path_to_optimizer_run_record_folder, algorithm_run['run_id']),
        'w+',
        encoding='utf-8'
    ) as file:
        file.write(txt_dump)
        file.flush()
        os.fsync(file)


# Beehive looks for minimum. Make this so that lowest value of this function means the best solution
@ignore_warnings(category=ConvergenceWarning)
def solution_evaluator(vector):
    global optimizer_iteration_counter, best_model

    optimizer_iteration_counter = optimizer_iteration_counter + 1

    if optimizer_iteration_counter % 2 == 0:
        temp = optimizer_iteration_counter // 2

        if temp % bee_count == 1:
            global iteration_start, iteration_end
            iteration_end = datetime.now()
            print('\tIteration (estimate) {}/{}, took {}. Best fitness: {}'.format(temp // bee_count + 1, iteration_cap, iteration_end - iteration_start, best_model['fitness']))
            iteration_start = iteration_end

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
                'args': split_dataset_master.Train.features[sorted_method_group_names[i]][temp_vector[temp_vector_starting_index + i]]['args'],
                'values': split_dataset_master.Train.features[sorted_method_group_names[i]][temp_vector[temp_vector_starting_index + i]]['values']
            })

    if len(inputs) < 2:
        return 2

    # Data for model
    dataset_fragments = FragmentedDatasetCV(inputs, split_dataset_master.Train.labels, cross_validation_fold_count)
    metric_values_test = []
    models = []
    for k in range(len(dataset_fragments.folds)):
        model_data = dataset_fragments.produce_split_dataset(k).produce_sklearn_ready_data()

        # ML Model
        model = model_type['model'](**param_dict)
        model.fit(model_data.train.features, model_data.train.labels)
        metric_value = fitness_metric['method'](model_data.test.labels, model.predict(model_data.test.features))
        if not math.isnan(metric_value):
            models.append(model)
            metric_values_test.append(metric_value)

    if len(metric_values_test) == 0:
        print('\tModel with NaN metric for all CV folds')
        return 2

    metric_avg_test = average(metric_values_test)
    average_model = models[min(range(len(models)), key=lambda x: abs(metric_avg_test - metric_values_test[x]))]

    fitness = 1 - metric_avg_test

    if best_model is None or fitness < best_model['fitness']:
        best_model = {
            'vector': vector,
            'inputs': [{'method_name': x['method_name'], 'args': x['args']} for x in inputs],
            'fitness': fitness,
            'hyperparams': param_dict,
            'model': average_model
        }

    return fitness


def run_optimization():
    # Create optimizer model
    model = Hive.BeeHive(
        lower=[0] * (len(arg_possibility_counts) + method_count),
        upper=list(map(lambda x: x - 1, arg_possibility_counts)) + list(map(lambda x: 2 * x - 1, method_param_counts)),
        fun=solution_evaluator,
        numb_bees=bee_count,
        max_itrs=iteration_cap
    )

    # Run da World
    optimizer_start = datetime.now()
    cost = model.run()
    optimizer_end = datetime.now()
    print('Optimization took: {}'.format(optimizer_end - optimizer_start))

    #Utilities.ConvergencePlot(cost)

    print("Best model:\n\tFitness: {}\n\tSolution: {}".format(best_model['fitness'], best_model['vector']))

    if 'models' not in algorithm_run['main'][key][dataset.name]:
        algorithm_run['main'][key][dataset.name]['models'] = {}

    # Persist the model
    model_id = model_id_generator.next_id()
    best_model['id'] = model_id

    global model_type
    best_model['type'] = model_type['name']

    global split_dataset_master
    best_model['validation'] = {
        'features': split_dataset_master.Validation.features,
        'labels': split_dataset_master.Validation.labels
    }
    dump(best_model['model'],  "./../resources/models/model_{}.jolib".format(model_id))
    del best_model['model']

    model_info_dump = json.dumps(best_model)
    with open('./../resources/models/model_{}.json'.format(best_model['id']), 'w+', encoding='utf-8') as file:
        file.write(model_info_dump)

    algorithm_run['main'][key][dataset.name]['models'][model_type['name']]['best_model_id'] = best_model['id']
    algorithm_run['main'][key][dataset.name]['models'][model_type['name']]['fitness_history'] = cost
    algorithm_run['main'][key][dataset.name]['models'][model_type['name']]['last_population'] = [x.vector for x in model.population]


load_optimizer_run(algorithm_run['run_id'])

dataset_counter = 1
dataset_counter_max = len(list(dataset_pool.keys())) * len(dataset_pool[list(dataset_pool.keys())[0]])

model_in_dataset_counter_max = len(model_types)

total_counter = 1
total_counter_max = dataset_counter_max * model_in_dataset_counter_max

global_start = datetime.now()
try:
    for key in dataset_pool:

        if key not in algorithm_run['main']:
            algorithm_run['main'][key] = {}

        for dataset in dataset_pool[key]:
            if dataset.name not in algorithm_run['main'][key]:
                algorithm_run['main'][key][dataset.name] = {}

            model_in_dataset_counter = 1
            for model_type in model_types:

                print('DATASET {}: {}/{} | MODEL {}: {}/{} | TOTAL: {}/{} {}%'.format(

                    dataset.name, dataset_counter, dataset_counter_max,
                    model_type['name'], model_in_dataset_counter, model_in_dataset_counter_max,
                    total_counter, total_counter_max, round(total_counter / total_counter_max, ndigits=4) * 100
                ))

                if 'models' not in algorithm_run['main'][key][dataset.name]:
                    algorithm_run['main'][key][dataset.name]['models'] = {}

                if model_type['name'] not in algorithm_run['main'][key][dataset.name]['models']:
                    print('DOES NOT EXIST - TRAINING')
                    algorithm_run['main'][key][dataset.name]['models'][model_type['name']] = {}

                    persisted_methods_temp = dataset.load_values()
                    gold_values_temp = [round(x / 5, ndigits=3) for x in
                                        persisted_methods_temp[gold_standard_name][0]['values']]
                    del persisted_methods_temp[gold_standard_name]

                    for method_name in persisted_methods_temp:
                        for i in range(len(persisted_methods_temp[method_name])):
                            if 'corpus' in persisted_methods_temp[method_name][i]['args']:
                                if key == 'lemma':
                                    persisted_methods_temp[method_name][i]['args']['corpus'] = persisted_methods_temp[method_name][i]['args']['corpus'].replace('_sk.txt', '_sk_lemma.txt')
                                else:
                                    persisted_methods_temp[method_name][i]['args']['corpus'] = persisted_methods_temp[method_name][i]['args']['corpus'].replace('_sk_lemma.txt', '_sk.txt')

                    split_dataset_master = FragmentedDatasetSuper(persisted_methods_temp, gold_values_temp,
                                                                  dataset_split_ratio)
                    sorted_method_group_names = sorted(persisted_methods_temp.keys())
                    method_count = len(sorted_method_group_names)

                    method_param_counts = [len(persisted_methods_temp[sorted_method_group_names[i]])
                                           for i in range(method_count)]

                    sorted_arg_names = sorted(model_type['args'].keys())
                    arg_possibility_counts = list(map(lambda x: len(model_type['args'][x]), sorted_arg_names))

                    if 'available_methods' not in algorithm_run['main'][key][dataset.name]:

                        possibilities = {method_name: [config['args'] for config in persisted_methods_temp[method_name]] for method_name in sorted_method_group_names}
                        algorithm_run['main'][key][dataset.name]['available_methods'] = possibilities

                    best_model = None
                    optimizer_iteration_counter = 0
                    iteration_start = datetime.now()
                    iteration_end = None

                    run_optimization()

                    persist_optimizer_run()

                else:
                    print('EXISTS - SKIPPING')

                model_in_dataset_counter = model_in_dataset_counter + 1
                total_counter = total_counter + 1

            dataset_counter = dataset_counter + 1

    playsound('./../sounds/victory.mp3')
    print('ENTIRE ALGORITHM TOOK {}'.format(datetime.now() - global_start))

except:
    playsound('./../sounds/wrong {}.mp3')
    print('ENTIRE ALGORITHM TOOK {}'.format(datetime.now() - global_start))
    raise sys.exc_info()[0]
