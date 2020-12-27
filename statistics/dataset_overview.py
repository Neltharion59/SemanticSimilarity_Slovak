import re
from os import listdir
from os.path import isfile, join

input_path = "./../resources/datasets/sts_processed/"

dataset_input_file_name_pattern = re.compile(".*_sk.txt")
sts_dataset_name_pattern = re.compile("sts_201[2-6]_[a-zA-Z]+]")

input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

for input_dataset_file in input_dataset_files:
    dataset_name = input_dataset_file.replace("_sk.txt", "").replace("dataset_", "")
    if sts_dataset_name_pattern.match(dataset_name):
        dataset_name = dataset_name.replace("_", " ")
    elif "sick" in input_dataset_file:
        dataset_name = "sick"

    with open(input_path + input_dataset_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        dataset_size = str(len(lines))
        dataset_number_type = "Celé čísla"
        for number in map(lambda x: x.split("\t")[0], lines):
            if "." in number:
                dataset_number_type = "Desatinné čísla"

        print("\t".join([dataset_name, dataset_size, dataset_number_type]))
