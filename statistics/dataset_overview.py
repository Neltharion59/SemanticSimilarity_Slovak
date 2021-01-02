# Runnable script calculating statistic values to be printed to console.
# Generate data in table form that can be imported to MS Excel so that it can be included in DP2 report.
# For each dataset, we have determine interesting properties.
# Row order: dataset_name, dataset size, STS value format (in Slovak)

import re
from os import listdir
from os.path import isfile, join

# Prepare path to read data from
input_path = "./../resources/datasets/sts_processed/"

# Prepare regexes to be used in this script
dataset_input_file_name_pattern = re.compile(".*_sk.txt")
sts_dataset_name_pattern = re.compile("sts_201[2-6]_[a-zA-Z]+]")

# Prepare list of dataset files to read data from
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

# Loop over all datasets, determine their properties and print them
for input_dataset_file in input_dataset_files:
    # Modify dataset name, so that it isn't too wide for A4 page
    dataset_name = input_dataset_file.replace("_sk.txt", "").replace("dataset_", "")
    if sts_dataset_name_pattern.match(dataset_name):
        dataset_name = dataset_name.replace("_", " ")
    elif "sick" in input_dataset_file:
        dataset_name = "sick"

    # Let's open the dataset and determine its size and STS value format
    with open(input_path + input_dataset_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # File size is simple
        dataset_size = str(len(lines))
        # Number format - based on whether there is a value with "." in it
        dataset_number_type = "Celé čísla"
        for number in map(lambda x: x.split("\t")[0], lines):
            if "." in number:
                dataset_number_type = "Desatinné čísla"

        print("\t".join([dataset_name, dataset_size, dataset_number_type]))
