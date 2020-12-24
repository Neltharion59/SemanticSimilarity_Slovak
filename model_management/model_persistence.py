from joblib import dump, load

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


# Params: str, STSModel, [bool]
# Return: None
def persist_sklearn_model(dataset, model, force_overwrite=False):

    # TODO check if model is already persisted
    # TODO Consider force_overwrite param

    try:
        with open(path_2_model_id_counter, "r", encoding='utf-8') as id_file:
            last_model_id = int(id_file.read())
    except FileNotFoundError:
        last_model_id = 0

    new_model_id = last_model_id + 1

    # Store the trained model
    dump(model.method, path_2_model_directory + "/model_" + str(new_model_id) + ".jolib")

    # Store new id
    with open(path_2_model_id_counter, "w+", encoding='utf-8') as id_file:
        id_file.write(str(new_model_id))

    # Store its description
    file_description = create_model_description(model, new_model_id, dataset) + "\n"

    with open(path_2_model_description_file, "a+", encoding='utf-8') as description_file:
        description_file.write(file_description)

    pass


# Params: int
# Return: STSModel
def load_sklearn_model(model_id):
    model = load(path_2_model_directory + "/model_" + str(model_id) + ".jolib")
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
def get_model_id(model):
    try:
        with open(path_2_model_description_file, "r", encoding='utf-8') as description_file:
            model_description_suffix = create_mode_description_suffix(model)
            for line in description_file:
                temp_line = ":" + ":".join(line.replace("\n", "").split(":")[2:])
                if temp_line == model_description_suffix:
                    model_id = int(line.split(":")[0])
                    return model_id
    except FileNotFoundError:
        return None

    return None


# Params: STSModel
# Return: bool
def model_exists(model):
    model_id = get_model_id(model)
    return model_id is not None
