from model_management.sts_method_wrappers import STSMethod

input_folder = "./../resources/datasets/sts_processed/"
output_folder = "./../resources/datasets/sts_method_values/"
gold_standard_name = "gold_standard"


def predict_and_persist_values(sts_method, dataset_name):
    words1, words2 = load_dataset(dataset_name)
    predicted_values = sts_method.predict_mass(words1, words2)
    persist_values(sts_method.name, dataset_name, predicted_values)


def persist_values(sts_method_name, dataset_name, values):
    output_file_path = output_folder + dataset_name
    persisted_method_types = get_persisted_method_types(dataset_name)
    if sts_method_name not in persisted_method_types:
        line = sts_method_name + ":" + ','.join(list(map(lambda x: str(x), values))) + '\n'

        with open(output_file_path, 'a+', encoding='utf-8') as output_file:
            output_file.write(line)


def persist_gold_standard(dataset_name):
    input_file_path = input_folder + dataset_name
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    values = list(map(lambda x: float(x.split('\t')[0]), lines))

    persist_values(gold_standard_name, dataset_name, values)


def get_persisted_method_types(dataset_name):
    input_file_path = output_folder + dataset_name
    method_list = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                method_name = line.split(':')[0]
                method_list.append(method_name)
    except FileNotFoundError:
        pass
    return method_list


def load_dataset(dataset_name):
    input_file_path = input_folder + dataset_name
    words1, words2 = [], []
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            tokens = line.split('\t')
            words1.append(tokens[1])
            words2.append(tokens[2])

    return words1, words2
