# Library-like script providing dataset wrapper class and few helpful variables and functions

import json
import os

# Prepare paths to work with in this scripts
input_folder = "./../resources/datasets/sts_processed/"
output_folder = "./../resources/datasets/sts_method_values/"

# Define name of gold standard so that this string is only hardcoded in one place
gold_standard_name = "gold_standard"


# Check if two dicts are exact value match (same keys and same values). Not nested.
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


# Wrapper class for dataset
class Dataset:
    # Constructor
    # Params: str, list<str>
    # Return: Dataset
    def __init__(self, name, dataset_names):
        self.name = name
        self.dataset_names = dataset_names

    # Load and return text pairs of given dataset and return them as two lists.
    # Params:
    # Return: list<str>, list<str>
    def load_dataset(self):
        words1, words2 = [], []

        # Load pairs of texts from disk
        for dataset_name in self.dataset_names:
            # Prepare path to value file to read from
            input_file_path = input_folder + dataset_name
            # Load texts from disk
            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                # Loop over lines of dataset file
                for line in input_file:
                    tokens = line.split('\t')
                    # Retrieve texts of current pair
                    words1.append(tokens[1])
                    words2.append(tokens[2])

        return words1, words2

    # Persist gold standard from dataset file to file with values.
    # Params:
    # Return:
    def persist_gold_standard(self):
        results = self.load_values()
        if gold_standard_name in results:
            return

        values = []
        for dataset_name in self.dataset_names:
            # Define the path to dataset file to read from
            input_file_path = input_folder + dataset_name
            # Read lines of the dataset file
            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                lines = input_file.readlines()
            # Retrieve STS values from lines of the file
            values = values + list(map(lambda x: float(x.split('\t')[0]), lines))

        results = {
            gold_standard_name: [
                {
                    'args': {},
                    'values': values
                }
            ]
        }
        # Save the values to file with values
        self.persist_values(results)

    # Predict values using given method for given dataset and persist those values
    # (only those that aren't persisted yet).
    # Params: STSMethod
    # Return:
    def predict_and_persist_values(self, sts_method):
        print("About to predict&persist values of {} method for {} dataset".format(sts_method.name, self.name))
        # Load the dataset from disk based on its name
        words1, words2 = self.load_dataset()

        results = self.load_values()

        if sts_method.method_name not in results:
            results[sts_method.method_name] = []

        for x in results[sts_method.method_name]:
            if dict_match(x['args'], sts_method.args):
                print("Already predicted. Skipping " + sts_method.name)
                return

        values = list(sts_method.predict_mass(words1, words2, {}))
        values = [round(x, ndigits=3) for x in values]
        print("values: {}".format(values))
        results[sts_method.method_name].append({
            'args': sts_method.args,
            'values': values
        })

        self.persist_values(results)
        print("Finished prediction")

    # Persist predicted values for given method for given dataset.
    # Params: dict<str, list<dict<str, ...>>>
    # Return: None
    def persist_values(self, results):
        # Prepare path to file to write to
        output_file_path = output_folder + self.name + ".txt"
        results_json = json.dumps(results)
        with open(output_file_path, 'w+', encoding='UTF-8') as file:
            file.write(results_json)
            file.flush()
            os.fsync(file)

    # Persist predicted values for given method for given dataset.
    # Params:
    # Return: dict<str, list<dict<str, ...>>>
    def load_values(self):
        # Prepare path to which to persist values to
        input_file_path = output_folder + self.name + ".txt"
        print(input_file_path)
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

    # Delete all values of given method for given dataset from the disk.
    # Useful if a method was bugged when its values were being persisted.
    # Params: STSMethod
    # Return: None
    def delete_values(self, sts_method):
        results = self.load_values()
        results = [x for method_name in results for x in results[method_name] if
                   not dict_match(sts_method.args, x['args']) or method_name != sts_method.method_name]
        self.persist_values(results)
