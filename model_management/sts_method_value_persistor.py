# Library-like script providing utility for persisting and loading values of models and methods
# Focused on sklearn models right now

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
    # See which method types are already persisted for given dataset
    persisted_method_types = get_persisted_method_types(dataset_name)
    persisted_values = []
    # If some values of the method are already persisted, let's make sure we only work with the new ones
    if sts_method.name in persisted_method_types:
        print("This dataset with this method already has persisted values")
        # See how many values for this dataset with this method are already persisted
        persisted_values = get_persisted_method_values(dataset_name)[sts_method.name]
        if len(persisted_values) >= len(words1):
            print("All values already predicted, no more action to be done")
            return
        print("Cutting down dataset size from {} by {} to {}".format(len(words1), len(persisted_values), abs(len(words1) - len(persisted_values))))
        # Cut the records with already calculated values
        words1 = words1[len(persisted_values):]
        words2 = words2[len(persisted_values):]

    print("Beginning prediction")
    # Initialize the record counter
    i, i_max = 1, len(words1)
    # Prepare path to which to persist values to
    output_file_path = output_folder + dataset_name

    # Let's open the file to append to
    with open(output_file_path, 'a+', encoding='utf-8') as output_file:
        # If there is no record for this method for this dataset yet, let's initialize it
        if len(persisted_values) == 0 and sts_method.name not in persisted_method_types:
            output_file.write(sts_method.name + ":")

        # Loop over all records that need to be predicted and persisted
        for predicted_value in sts_method.predict_mass(words1, words2, sts_method_pool):
            print("Predicting sample number {}/{} - {}%".format(i, i_max, round(i/i_max, 2) * 100))
            output_text = ""
            # Prepend value with delimeter, if it is not the first one
            if i > 1 or len(persisted_values) > 0:
                output_text = ","
            # Append the value to resulting line
            output_text = output_text + str(predicted_value)
            # Append the line to file
            output_file.write(output_text)

            # Make sure that file mofications are persisted on disk at least every 100 records
            if i % 100 == 0:
                output_file.flush()
            # Increment record counter
            i = i + 1

        # If we are done and persisted at least one value, let's add newline char
        if i > 1:
            output_file.write("\n")

    print("Finishing prediction")


# Persist predicted values for given method for given dataset.
# Params: str, str, list<float>
# Return:
def persist_values(sts_method_name, dataset_name, values):
    # Prepare path to file to write to
    output_file_path = output_folder + dataset_name
    # See which methods already have persisted values for given dataset
    persisted_method_types = get_persisted_method_types(dataset_name)
    # If our method has no values persisted yet, let's append them to file
    if sts_method_name not in persisted_method_types:
        # Create new entry - new line - in the file
        line = sts_method_name + ":" + ','.join(list(map(lambda x: str(x), values))) + '\n'
        # Write the line to file on disk
        with open(output_file_path, 'a+', encoding='utf-8') as output_file:
            output_file.write(line)
    # If method already has persisted values, but isn't gold standard,
    # let's see if we need to write some of them on the disk
    elif sts_method_name != 'gold_standard':
        # Concant our values to already persisted values
        persisted_values = get_persisted_method_values(dataset_name)[sts_method_name]
        values_to_persist = persisted_values + values
        # Get all the lines from persisted values
        with open(output_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
        # Loop over the lines and append values to where they belong
        for i in range(len(lines)):
            line_method_name = lines[i].split(":")[0]
            if line_method_name == sts_method_name:
                lines[i] = line_method_name + ":" + ",".join(values_to_persist) + "\n"
                with open(output_file_path, 'w+', encoding='utf-8') as output_file:
                    output_file.writelines(lines)
                break


# Persist gold standard from dataset file to file with values.
# Params: str
# Return:
def persist_gold_standard(dataset_name):
    # Define the path to dataset file to read from
    input_file_path = input_folder + dataset_name
    # Read lines of the dataset file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    # Retrieve STS values from lines of the file
    values = list(map(lambda x: float(x.split('\t')[0]), lines))
    # Save the values to file with values
    persist_values(gold_standard_name, dataset_name, values)


# Get list of method names of all methods that have some values persisted for given dataset.
# Params: Dataset
# Return: list<str>
def get_persisted_method_types(dataset):
    method_lists = []
    for dataset_name in dataset.dataset_names:
        # Get path to value file to read from
        input_file_path = output_folder + dataset_name
        method_list = []
        # Read method names from the file
        try:
            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    method_name = line.split(':')[0]
                    method_list.append(method_name)
        except FileNotFoundError:
            pass
        method_lists.append(method_list)

    result = set(method_lists[0])
    for method_list in method_lists[1:]:
        result = result & set(method_list)
    result = list(result)

    return result


# Get dict with method names being keys to lists of their values
# Params: str
# Return: dict<str, list<float>>
def get_persisted_method_values(dataset_name):
    # Prepare path to value file to read from
    input_file_path = output_folder + dataset_name

    # Read method names and values from the file
    method_pool = {}
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            # Loop over lines of file
            for line in input_file:
                line = line.replace('\n', '')

                # Empty lines are not interesting
                if len(line) == 0:
                    continue

                # Retrieve method name and values from current lines
                method_name = line.split(':')[0]
                method_values = [] if len(line.split(':')[1]) < 1 else list(map(lambda x: float(x), line.split(':')[1].split(',')))
                # Append retrieved info to resulting dict
                method_pool[method_name] = method_values

    except FileNotFoundError:
        return None

    return method_pool


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
# Params: str, str
# Return: bool
def delete_values(dataset_name, method_name):
    # Prepare path to value file to read from
    dataset_values_file_path = output_folder + dataset_name
    # Read the file content from the disk
    try:
        with open(dataset_values_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return False

    # Modify the transient content of the file
    new_lines = []
    # Loop over lines of transient file
    for i in range(len(lines)):
        # Retrieve method name of current line
        current_method_name = lines[i].split(":")[0]
        # If current method is the one we want to delete values of, we simply do not append this line to new file
        if current_method_name == method_name:
            continue

        # If current method is not the one we want to delete, let's append the current line to new file
        new_lines.append(lines[i])

    # Write the new file to disk, overwriting the new one
    with open(dataset_values_file_path, 'w+', encoding='utf-8') as file:
        file.writelines(new_lines)

    return True
