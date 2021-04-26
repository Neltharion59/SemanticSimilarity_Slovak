# Library-like script providing utility for persisting and loading values of models and methods
# Focused on sklearn models right now
import json
import os

from model_management.sts_method_pool import sts_method_pool

# Prepare paths to work with in this scripts
input_folder = "./../resources/datasets/sts_processed/"
output_folder = "./../resources/datasets/sts_method_values/"

# Define name of gold standard so that this string is only hardcoded in one place
gold_standard_name = "gold_standard"


# Predict values using given method for given dataset and persist those values (only those that aren't persisted yet).
# Params: STSMethod, str
# Return:
def predict_and_persist_values(sts_method, dataset_name):
    print("About to predict&persist values of {} method for {} dataset".format(sts_method.name, dataset_name))
    # Load the dataset from disk based on its name
    words1, words2 = load_dataset(dataset_name)

    results = load_values(dataset_name)

    if sts_method.method_name not in results:
        results[sts_method.method_name] = []

    for x in results[sts_method.method_name]:
        if dict_match(x['args'], sts_method.args):
            print("Already predicted. Skipping " + sts_method.name)
            return

    values = list(sts_method.predict_mass(words1, words2, {}))

    results[sts_method.method_name].append({
        'args': sts_method.args,
        'values': values
    })

    persist_values(results, dataset_name)
    print("Finished prediction")


# Check if two dicts are exact value match
# Params: dict, dict
# Return: bool
def dict_match(dict1, dict2):
    for x in dict1:
        if x not in dict2 or dict1[x] != dict2[x]:
            return False
    for x in dict2:
        if x not in dict1 or dict1[x] != dict2[x]:
            return False
    return True


# Persist predicted values for given method for given dataset.
# Params: dict<str, list<dict<str, ...>>>, str
# Return: None
def persist_values(results, dataset_name):
    # Prepare path to file to write to
    output_file_path = output_folder + dataset_name
    results_json = json.dumps(results, sort_keys=True, indent=4)
    with open(output_file_path, 'w+', encoding='UTF-8') as file:
        file.write(results_json)
        file.flush()
        os.fsync(file)


# Persist predicted values for given method for given dataset.
# Params: str
# Return: dict<str, list<dict<str, ...>>>
def load_values(dataset_name):
    # Prepare path to which to persist values to
    input_file_path = output_folder + dataset_name

    # Let's open the file to append to
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            print("file found")
            text = file.read()
        results = json.loads(text)
    except FileNotFoundError:
        print("file not found")
        results = {}

    return results


# Persist gold standard from dataset file to file with values.
# Params: str
# Return:
def persist_gold_standard(dataset_name):
    results = load_values(dataset_name)
    if gold_standard_name in results:
        return

    # Define the path to dataset file to read from
    input_file_path = input_folder + dataset_name
    # Read lines of the dataset file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    # Retrieve STS values from lines of the file
    values = list(map(lambda x: float(x.split('\t')[0]), lines))

    results = {
        gold_standard_name: [
            {
                'args': {},
                'values': values
            }
        ]
    }
    # Save the values to file with values
    persist_values(results, dataset_name)


# Load and return text pairs of given dataset and return them as two lists.
# Params: str
# Return: list<str>, list<str>
def load_dataset(dataset_name):
    # Prepare path to value file to read from
    input_file_path = input_folder + dataset_name

    # Load pairs of texts from disk
    words1, words2 = [], []
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        # Loop over lines of dataset file
        for line in input_file:
            tokens = line.split('\t')
            # Retrieve texts of current pair
            words1.append(tokens[1])
            words2.append(tokens[2])

    return words1, words2


# Delete all values of given method for given dataset from the disk.
# Useful if a method was bugged when its values were being persisted.
# Params: str, STSMethod
# Return: None
def delete_values(dataset_name, sts_method):
    results = load_values(dataset_name)

    results = [x for method_name in results for x in results[method_name] if not dict_match(sts_method.args, x['args']) or method_name != sts_method.method_name]

    persist_values(results, dataset_name)

