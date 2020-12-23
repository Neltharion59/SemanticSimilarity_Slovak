import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model_management.sts_method_value_persistor import get_persisted_method_values, gold_standard_name
from evaluation.evaluate_regression_metrics import evaluate_prediction_metrics


def prepare_training_data(dataset, methods):
    available_values = get_persisted_method_values(dataset)
    removal_keys = []
    for key in available_values:
        if key not in methods and key != gold_standard_name:
            removal_keys.append(key)
    for key in removal_keys:
        available_values.pop(key, None)
    available_values[gold_standard_name] = list(map(lambda x: x/5.0, available_values[gold_standard_name]))
    table = []
    for key in available_values:
        print("{}: {} values".format(key, len(available_values[key])))
        table.append(available_values[key])

    numpy_array = np.array(table)
    numpy_array = numpy_array.transpose()

    df = pd.DataFrame(numpy_array, index=range(1, numpy_array.shape[0] + 1), columns=list(map(lambda x: x.split("___")[0], available_values.keys())))
    print(df.head(10))

    train, test = train_test_split(df, test_size=0.3)

    x_train = train.drop(gold_standard_name, axis=1)
    x_test = test.drop(gold_standard_name, axis=1)
    y_train = train[gold_standard_name]
    y_test = test[gold_standard_name]

    return x_train, x_test, y_train, y_test


def train_n_test(x_train, x_test, y_train, y_test, model):
    print("Training begins")
    model.fit(x_train, y_train)
    print("Training done")
    print("Prediction begins")
    lr_score = model.predict(x_test)
    print("Prediction done")

    y_pred = model.predict(x_test)

    evaluation = evaluate_prediction_metrics(y_test, y_pred, 1)
    for key in evaluation:
        print("{}: {}".format(key, evaluation[key]))
