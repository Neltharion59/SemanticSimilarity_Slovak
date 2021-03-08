# Library-like script providing functions for aggregation methods

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_management.sts_method_value_persistor import get_persisted_method_values, gold_standard_name
from evaluation.evaluate_regression_metrics import evaluate_prediction_metrics
from dataset_modification_scripts.dataset_wrapper import Dataset


# Split given dataset (identified by name) to training and testing data, and to attributes and labels
# Used to prepare data for aggregation models.
# Params: str, list<str>
# Return: DataFrame, DataFrame, DataFrame, DataFrame
def prepare_training_data(dataset, methods, print_head=True):
    if not isinstance(dataset, Dataset):
        raise ValueError("Expected instance of Dataset class, got {} instead".format(type(dataset)))

    # Load all persisted values of given dataset
    available_values = dataset.get_data()

    # Let's remove all values from the loaded dict of values which we aren't gonna use
    removal_keys = []
    for key in available_values:
        if key not in methods and key != gold_standard_name:
            removal_keys.append(key)
    for key in removal_keys:
        available_values.pop(key, None)

    # Let's make sure the gold standard is normalized into the same range of values as other methods
    available_values[gold_standard_name] = list(map(lambda x: x/5.0, available_values[gold_standard_name]))

    # Let's prepare table of values for the model (and print the sizes of values in case some are missing)
    table = []
    for key in available_values:
        if print_head:
            print("{}: {} values".format(key, len(available_values[key])))
        table.append(available_values[key])

    # Let's turn the table to numpy array and transpose it (so that one row represents values for one pair of words)
    numpy_array = np.array(table)
    numpy_array = numpy_array.transpose()

    # Let's turn the table to DataFrame and print the head - make sure something didn't go wrong
    df = pd.DataFrame(numpy_array, index=range(1, numpy_array.shape[0] + 1), columns=list(map(lambda x: x.split("___")[0], available_values.keys())))
    if print_head:
        print(df.head(10))

    # Let's split the dataset to testing and training subsets (using sklearn function)
    train, test = train_test_split(df, test_size=0.3)

    # Let's prepare testing and training attribute values and labels
    x_train = train.drop(gold_standard_name, axis=1)
    x_test = test.drop(gold_standard_name, axis=1)
    y_train = train[gold_standard_name]
    y_test = test[gold_standard_name]

    return x_train, x_test, y_train, y_test


# Train given sklearn model using training data and evaluate it using testing data.
# Return dict of metrics of trained model on testing data
# Params: DataFrame, DataFrame, DataFrame, DataFrame, sklearn.model
# Return: dict
def train_n_test(x_train, x_test, y_train, y_test, model):
    # Train the model
    print("Training begins")
    model.fit(x_train, y_train)
    print("Training done")

    # Test the model
    print("Prediction begins")
    y_pred = model.predict(x_test)
    print("Prediction done")

    # Evaluate metrics of model on testing dataset
    evaluation = evaluate_prediction_metrics(y_test, y_pred, 1)

    # Print those metrics
    for key in evaluation:
        print("{}: {}".format(key, evaluation[key]))

    return evaluation
