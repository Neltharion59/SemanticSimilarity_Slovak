import re
from os import listdir
from os.path import isfile, join

working_path = "./../resources/datasets/sts_method_values/"

dataset_input_file_name_pattern = re.compile("dataset_.*.txt")
model_name_pattern = re.compile("model_[0-9]+\:.*")

# Let's prepare list of dataset files to be lemmatized.s
input_dataset_files = [x for x in listdir(working_path) if isfile(join(working_path, x)) and dataset_input_file_name_pattern.match(x)]

for file_name in input_dataset_files:
    file_path = working_path + file_name

    with open(file_path, 'r', encoding='UTF-8') as current_file:
        lines = current_file.readlines()
    new_lines = list(filter(lambda x: not model_name_pattern.match(x), lines))
    print("{} down to {}".format(len(lines), len(new_lines)))
    with open(file_path, 'w', encoding='UTF-8') as current_file:
        current_file.writelines(new_lines)
