from model_management.sts_method_pool import sts_method_pool

input_folder = "./../resources/datasets/sts_processed/"
output_folder = "./../resources/datasets/sts_method_values/"
gold_standard_name = "gold_standard"


def predict_and_persist_values(sts_method, dataset_name):
    print("About to predict&persist values of {} method for {} dataset".format(sts_method.name, dataset_name))
    words1, words2 = load_dataset(dataset_name)
    persisted_method_types = get_persisted_method_types(dataset_name)
    persisted_values = []
    if sts_method.name in persisted_method_types:
        print("This dataset with this method already has persisted values")
        persisted_values = get_persisted_method_values(dataset_name)[sts_method.name]
        if len(persisted_values) >= len(words1):
            print("All values already predicted, no more action to be done")
            return
        print("Cutting down dataset size from {} by {} to {}".format(len(words1), len(persisted_values), abs(len(words1) - len(persisted_values))))
        words1 = words1[len(persisted_values):]
        words2 = words2[len(persisted_values):]

    print("Beginning prediction")
    i, i_max = 1, len(words1)
    output_file_path = output_folder + dataset_name

    with open(output_file_path, 'a+', encoding='utf-8') as output_file:
        if len(persisted_values) == 0 and sts_method.name not in persisted_method_types:
            output_file.write(sts_method.name + ":")

        for predicted_value in sts_method.predict_mass(words1, words2, sts_method_pool):
            print("Predicting sample number {}/{} - {}%".format(i, i_max, round(i/i_max, 2) * 100))
            output_text = ""
            if i > 1 or len(persisted_values) > 0:
                output_text = ","

            output_text = output_text + str(predicted_value)
            output_file.write(output_text)
            if i % 100 == 0:
                output_file.flush()
            i = i + 1
        if i > 1:
            output_file.write("\n")

    print("Finishing prediction")


def persist_values(sts_method_name, dataset_name, values):
    output_file_path = output_folder + dataset_name
    persisted_method_types = get_persisted_method_types(dataset_name)
    if sts_method_name not in persisted_method_types:
        line = sts_method_name + ":" + ','.join(list(map(lambda x: str(x), values))) + '\n'
        with open(output_file_path, 'a+', encoding='utf-8') as output_file:
            output_file.write(line)
    elif sts_method_name != 'gold_standard':
        persisted_values = get_persisted_method_values(dataset_name)[sts_method_name]
        values_to_persist = persisted_values + values
        with open(output_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
        for i in range(len(lines)):
            line_method_name = lines[i].split(":")[0]
            if line_method_name == sts_method_name:
                lines[i] = line_method_name + ":" + ",".join(values_to_persist) + "\n"
                with open(output_file_path, 'w+', encoding='utf-8') as output_file:
                    output_file.writelines(lines)
                break


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


def get_persisted_method_values(dataset_name):
    input_file_path = output_folder + dataset_name
    method_pool = {}
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                line = line.replace('\n', '')
                if len(line) == 0:
                    continue
                method_name = line.split(':')[0]
                method_values = [] if len(line.split(':')[1]) < 1 else list(map(lambda x: float(x), line.split(':')[1].split(',')))
                method_pool[method_name] = method_values
    except FileNotFoundError:
        return None
    return method_pool


def load_dataset(dataset_name):
    input_file_path = input_folder + dataset_name
    words1, words2 = [], []
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            tokens = line.split('\t')
            words1.append(tokens[1])
            words2.append(tokens[2])

    return words1, words2
