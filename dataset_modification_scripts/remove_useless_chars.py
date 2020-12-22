from os import listdir
from os.path import isfile, join
import re

input_path = "./../resources/datasets/sts_processed/"
output_path = "./../resources/datasets/sts_processed/"

dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?.txt")

input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

i = 1
total_count = len(input_dataset_files)
for input_dataset_file in input_dataset_files:
    output_dataset_file = output_path + input_dataset_file

    existing_output_file_lines = []
    try:
        with open(output_dataset_file, 'r', encoding='utf-8') as output_file:
            existing_output_file_text = output_file.read()
    except FileNotFoundError:
        pass

    existing_output_file_text = existing_output_file_text.replace(",", "")
    existing_output_file_text = existing_output_file_text.replace("? ", "")
    existing_output_file_text = existing_output_file_text.replace(" ?", "")

    with open(output_dataset_file, "w+", encoding='utf-8') as output_file:
        output_file.write(existing_output_file_text)

    print("{} - {}/{} - {:.2%}".format(output_dataset_file, i, total_count, i/total_count))
    i = i + 1
