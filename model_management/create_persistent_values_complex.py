# Runnable script calculating values for each dataset for each aggregation method and persisting them
# Already persisted values are not calculated again nor persisted

from os import getcwd
import sys

# Mandatory if we want to run this script from windows cmd. Must precede all imports from this project
from complex_similarity_methods.regression_methods_core import prepare_training_data

conf_path = getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/..')
sys.path.append(conf_path + '/../..')

from model_management.sts_model_pool import sts_model_pool
from model_management.sts_method_value_persistor import predict_and_persist_values, get_persisted_method_types,\
    delete_values
from model_management.model_persistence import persist_sklearn_model, get_model_id, load_sklearn_model,\
    delete_sklearn_model

# If values are already calculated, should we calculate them again and overwrite persisted values
recalculate_values = False

# Loop over each dataset and calculate values for aggregation methods
for dataset in sts_model_pool:
    # Print to screen info of which dataset we are processing right now
    print("Dataset {}".format(dataset))

    # See which methods already have at least some values persisted for them
    persisted_methods = get_persisted_method_types(dataset)

    # Loop over each aggregation method for this dataset
    for model_name in sts_model_pool[dataset]:
        # Print to screen which aggregation method we are calculating right now
        print("Dataset {}, model {}".format(dataset, sts_model_pool[dataset][model_name].name))

        # If the model already exists, let's get its id
        existing_model_id = get_model_id(dataset, sts_model_pool[dataset][model_name])

        # 1. Delete existing model if needed
        print("Checking if model needs to be deleted.")
        if recalculate_values and existing_model_id is not None:
            print("Need to delete the model.")
            delete_sklearn_model(existing_model_id)
            print("Model deleted.")
        else:
            print("No need to delete the model.")

        # 2. Delete existing values
        print("Checking if existing calculated values need to be deleted.")
        if recalculate_values and model_name in persisted_methods:
            print("Need to delete values.")
            delete_values(dataset, "model_" + str(existing_model_id))
            print("Values deleted.")
        else:
            print("No need to delete values.")

        # 3. Train model
        print("Checking if model needs to be trained.")
        if recalculate_values or existing_model_id is None:
            print("About to train the model as it is not trained yet.")
            x_train, x_test, y_train, y_test = prepare_training_data(
                dataset,
                sts_model_pool[dataset][model_name].input_method_names
            )
            train_metrics = sts_model_pool[dataset][model_name].train(x_train, x_test, y_train, y_test)
            print("Model trained")
        else:
            print("No need to train the model.")

        # 4. Save model
        print("Checking if model needs to be saved.")
        if recalculate_values or existing_model_id is None:
            print("Need to save the model.")
            persist_sklearn_model(dataset, sts_model_pool[dataset][model_name], train_metrics, force_overwrite=True)
            print("Model saved")
        else:
            print("No need to save the model.")

        # If we created a new persisted model, let's get its id
        if existing_model_id is None:
            existing_model_id = get_model_id(dataset, sts_model_pool[dataset][model_name])
            if existing_model_id is None:
                print("The {} model has no id. Something went wrong. Shutting down.".format(sts_model_pool[dataset][model_name].name))
                exit()

        # If we have untrained existing model, let's load its persisted form
        print("Checking if model needs to be loaded.")
        if not sts_model_pool[dataset][model_name].trained:
            print("Need to load the model.")
            sts_model_pool[dataset][model_name].method = load_sklearn_model(existing_model_id)
            sts_model_pool[dataset][model_name].trained = True
            print("Model loaded")
        else:
            print("No need to load the model.")

        # 5. Predict and persist values
        print("About to predict and save values for this model.")
        temp_name = sts_model_pool[dataset][model_name].name
        sts_model_pool[dataset][model_name].name = "model_" + str(existing_model_id)
        predict_and_persist_values(sts_model_pool[dataset][model_name], dataset)
        sts_model_pool[dataset][model_name].name = temp_name
        print("Values predicted and saved.")

        print("{} processed in the end.".format(sts_model_pool[dataset][model_name].name))
