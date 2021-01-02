# Library-like script providing utility for persisting and loading models and their info
# Focused on sklearn models right now

from joblib import dump, load
from os import remove
from complex_similarity_methods.regression_methods_core import train_n_test
from model_management.sts_method_wrappers import STSModel

# Paths to directories we are going to use in this script
path_2_model_id_counter = "./../resources/model_id_counter.txt"
path_2_model_directory = "./../resources/models"
path_2_model_description_file = "./../resources/model_descriptions.txt"


# Create suffix of description of the model.
# Params: STSModel
# Return: str
def create_mode_description_suffix(model):
    file_description_suffix = ":" + model.name + ":"

    # Include names of input methods
    for i in range(len(model.input_method_names)):
        if i > 0:
            file_description_suffix = file_description_suffix + ","
        file_description_suffix = file_description_suffix + model.input_method_names[i]

    return file_description_suffix


# Create description of the model.
# Params: STSModel, int, str
# Return: str
def create_model_description(model, model_id, dataset_name):
    file_description = str(model_id) + ":" + dataset_name + create_mode_description_suffix(model)
    return file_description


# Create filename for persisted model form.
# Params: int
# Return: str
def create_model_file_name(model_id):
    return path_2_model_directory + "/model_" + str(model_id) + ".jolib"


# Delete persisted sklearn model with given id.
# Params: int
# Return:
def delete_sklearn_model(model_id):
    model_file_name = create_model_file_name(model_id)
    remove(model_file_name)


# Create metric description of the model.
# Params: dict
# Return: str
def create_metric_description(metric_dict):
    result = ""
    i = 1
    # For each metric, append its name and value to description
    for metric in metric_dict:
        if i > 1:
            result = result + ","
        if isinstance(metric_dict[metric], tuple):
            metric_dict[metric] = metric_dict[metric][0]
        result = result + metric + "-" + str(metric_dict[metric])
        i = i + 1

    return result


# Persist sklearn model on the disk with its custom id.
# Params: str, STSModel, dict, [bool]
# Return: bool
def persist_sklearn_model(dataset, model, train_metrics, force_overwrite=False):

    # Check if model is already persisted
    existing_model_id = get_model_id(dataset, model)
    # If it is already persisted
    if existing_model_id is not None:
        print("Such model is already persisted, with id {}".format(existing_model_id))
        # and we do not want a new one, time to quit
        if not force_overwrite:
            print("We do not desire to overwrite it, so we quit")
            return False
        # and we want a new one, let's delete it first
        else:
            print("We desire to overwrite it, so we are deleting the model (but preserve its id)")
            delete_sklearn_model(existing_model_id)
            pass

    # If the model is not persisted yet, let's generate it a new id
    if existing_model_id is None:
        try:
            with open(path_2_model_id_counter, "r", encoding='utf-8') as id_file:
                last_model_id = int(id_file.read())
        except FileNotFoundError:
            last_model_id = 0

        new_model_id = last_model_id + 1
    else:
        new_model_id = existing_model_id

    # Store the trained model
    dump(model.method, path_2_model_directory + "/model_" + str(new_model_id) + ".jolib")

    if existing_model_id is None:
        # Store new id
        with open(path_2_model_id_counter, "w+", encoding='utf-8') as id_file:
            id_file.write(str(new_model_id))

        # Store its description
        file_description = create_model_description(model, new_model_id, dataset) + ":" + create_metric_description(train_metrics) +"\n"

        with open(path_2_model_description_file, "a+", encoding='utf-8') as description_file:
            description_file.write(file_description)

    print("Model successfully persisted with id {}".format(new_model_id))

    return True


# Load persisted sklearn model from the disk based on its id.
# Params: int
# Return: sklearn.model
def load_sklearn_model(model_id):
    return load(path_2_model_directory + "/model_" + str(model_id) + ".jolib")


# Load persisted sklearn model from the disk based on its id and wrap it in our wrapper class STSModel.
# Params: int
# Return: STSModel
def load_sklearn_model_wrapper(model_id):
    # Load sklearn model from the disk using its id.
    model = load_sklearn_model(model_id)
    found = False

    # Try to open the file with model list and description
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            # Loop over all lines
            for line in description_file:
                line = line.replace("\n", "")
                # Split the line to tokens (1 token - 1 attribute of model)
                tokens = line.split(":")
                current_model_id = int(tokens[0])

                # If current model is the one we were looking for,
                # retrieve its info
                if current_model_id == model_id:
                    dataset_name = tokens[1]
                    model_name = tokens[2]
                    input_names = tokens[3].split(',')
                    found = True
                    break
    except FileNotFoundError:
        return None

    if not found:
        return None

    # Let's wrap the model in our wrapper class STSModel
    wrapped_model = STSModel(model_name.split("___")[0], model, {}, input_names, train_n_test)
    wrapped_model.name = model_name
    wrapped_model.trained = True

    return wrapped_model


# Get id of model based on its attributes.
# Params: STSModel
# Return: int/None
def get_model_id(dataset, model):
    # Open the file with model descriptions
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            model_description_suffix = create_mode_description_suffix(model)
            # Loop over lines of file
            for line in description_file:
                dataset_name = line.split(":")[1]

                # If current model is trained for different dataset than we want, let's skip to another one
                if dataset_name != dataset:
                    continue

                # Modify the line to easily retrieve info
                temp_line = ":" + ":".join(line.replace("\n", "").split(":")[2:-1])

                # If we have found the model we are looking for, let's get the id of the model and quit
                if temp_line == model_description_suffix:
                    model_id = int(line.split(":")[0])
                    return model_id

    except FileNotFoundError:
        return None

    return None


# Get id of model based on its name and name of associated dataset.
# Params: str, str
# Return: int/None
def get_model_id_by_name(dataset, model_name):
    # Open the file with model description
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            # Loop over lines of the file
            for line in description_file:
                # Split model info into tokens (1 token - 1 attribute)
                tokens = line.split(":")
                dataset_name = tokens[1]

                # If the current dataset is not the one we want, let's skip to another model
                if dataset_name != dataset:
                    continue

                # If name of current model is the one we want, let's return its id
                model_name_current = tokens[2]
                if model_name_current == model_name:
                    model_id = int(tokens[0])
                    return model_id
    except FileNotFoundError:
        return None

    return None


# Check whether persistent model exists based on its name and name of associated dataset.
# Params: str, STSModel
# Return: bool
def model_exists(dataset, model):
    model_id = get_model_id(dataset, model)
    return model_id is not None


# Get model description based on its id
# Params: int
# Return: str
def get_model_description(model_id):
    # Load model description file
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            # Loop over all lines of the file
            for line in description_file:
                line = line.replace("\n", "")

                # Split the line to tokens (1 token - 1 attribute)
                tokens = line.split(":")
                current_model_id = int(tokens[0])

                # If id of current model is the one we were looking for, return model's description
                if current_model_id == model_id:
                    description = tokens[2]
                    return description

    except FileNotFoundError:
        pass

    return None


# Get metrics of persisted model with given id on testing data.
# Params: int
# Return: dict
def get_model_test_metrics(model_id):
    # Load model description file
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            # Loop over all lines of the file
            for line in description_file:
                line = line.replace("\n", "")

                # Split the line to tokens (1 token - 1 attribute)
                tokens = line.split(":")

                # If id of current model is the one we were looking for, return model's metrics
                current_model_id = int(tokens[0])
                if current_model_id == model_id:
                    metrics = tokens[4]
                    metrics = metrics.split(",")
                    result_dict = {}
                    for metric in metrics:
                        tokens = metric.split("-", 1)
                        result_dict[tokens[0]] = float(tokens[1])
                    return result_dict
    except FileNotFoundError:
        pass

    return None
