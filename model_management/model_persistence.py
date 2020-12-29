from joblib import dump, load
from os import remove
from complex_similarity_methods.regression_methods_core import train_n_test
from model_management.sts_method_wrappers import STSModel

path_2_model_id_counter = "./../resources/model_id_counter.txt"
path_2_model_directory = "./../resources/models"
path_2_model_description_file = "./../resources/model_descriptions.txt"


# Params: STSModel
# Return: str
def create_mode_description_suffix(model):
    file_description_suffix = ":" + model.name + ":"
    for i in range(len(model.input_method_names)):
        if i > 0:
            file_description_suffix = file_description_suffix + ","
        file_description_suffix = file_description_suffix + model.input_method_names[i]

    return file_description_suffix


# Params: STSModel, int, str
# Return: str
def create_model_description(model, model_id, dataset_name):
    file_description = str(model_id) + ":" + dataset_name + create_mode_description_suffix(model)
    return file_description


# Params: int
# Return: str
def create_model_file_name(model_id):
    return path_2_model_directory + "/model_" + str(model_id) + ".jolib"


# Params: int
# Return: str
def delete_sklearn_model(model_id):
    model_file_name = create_model_file_name(model_id)
    remove(model_file_name)


def create_metric_description(metric_dict):
    print(metric_dict)
    result = ""
    i = 1
    for metric in metric_dict:
        if i > 1:
            result = result + ","
        if isinstance(metric_dict[metric], tuple):
            metric_dict[metric] = metric_dict[metric][0]
        result = result + metric + "-" + str(metric_dict[metric])
        i = i + 1
    return result


# Params: str, STSModel, [bool]
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


# Params: int
# Return: sklearn.model
def load_sklearn_model(model_id):
    return load(path_2_model_directory + "/model_" + str(model_id) + ".jolib")


# Params: int
# Return: STSModel
def load_sklearn_model_wrapper(model_id):
    model = load_sklearn_model_wrapper(model_id)
    found = False
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            for line in description_file:
                line = line.replace("\n", "")
                tokens = line.split(":")
                current_model_id = int(tokens[0])
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

    wrapped_model = STSModel(model_name.split("___")[0], model, {}, input_names, train_n_test)
    wrapped_model.name = model_name
    wrapped_model.trained = True

    return wrapped_model


# Params: STSModel
# Return: int/None
def get_model_id(dataset, model):
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            model_description_suffix = create_mode_description_suffix(model)
            for line in description_file:
                dataset_name = line.split(":")[1]
                if dataset_name != dataset:
                    continue
                temp_line = ":" + ":".join(line.replace("\n", "").split(":")[2:-1])
                if temp_line == model_description_suffix:
                    model_id = int(line.split(":")[0])
                    return model_id
    except FileNotFoundError:
        return None

    return None


# Params: str, str
# Return: int/None
def get_model_id_by_name(dataset, model_name):
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            for line in description_file:
                tokens = line.split(":")
                dataset_name = tokens[1]
                if dataset_name != dataset:
                    continue
                model_name_current = tokens[2]
                if model_name_current == model_name:
                    model_id = int(tokens[0])
                    return model_id
    except FileNotFoundError:
        return None

    return None


# Params: STSModel
# Return: bool
def model_exists(dataset, model):
    model_id = get_model_id(dataset, model)
    return model_id is not None


def get_model_description(model_id):
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            for line in description_file:
                line = line.replace("\n", "")
                tokens = line.split(":")
                current_model_id = int(tokens[0])
                if current_model_id == model_id:
                    description = tokens[2]
                    return description
    except FileNotFoundError:
        pass

    return None


def get_model_test_metrics(model_id):
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            for line in description_file:
                line = line.replace("\n", "")
                tokens = line.split(":")
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
