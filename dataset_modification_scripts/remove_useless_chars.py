# Runnable script modifying datasets to not contain useless artifacts

from os import listdir
from os.path import isfile, join
import re

# Let's prepare the paths to read from and write to.
input_path = "./../resources/datasets/sts_processed/"
output_path = "./../resources/datasets/sts_processed/"

# Let's prepare regexes to be used throughout this script.
dataset_input_file_name_pattern = re.compile(".*_sk(_lemma)?.txt")

# Let's prepare list of dataset files to be modified.
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

# Initialize dataset file counter
i = 1
total_count = len(input_dataset_files)

# Let's loop over all input dataset files
for input_dataset_file in input_dataset_files:
    # Prepare full name of output file
    output_dataset_file = output_path + input_dataset_file

    # Let's load the file
    existing_output_file_lines = []
    try:
        with open(output_dataset_file, 'r', encoding='utf-8') as output_file:
            existing_output_file_text = output_file.read()
    except FileNotFoundError:
        pass

    # Apply modifications to the file
    existing_output_file_text = existing_output_file_text.replace(",", "")
    existing_output_file_text = existing_output_file_text.replace("? ", "")
    existing_output_file_text = existing_output_file_text.replace(" ?", "")

    # Overwrite the original with its modified version
    with open(output_dataset_file, "w+", encoding='utf-8') as output_file:
        output_file.write(existing_output_file_text)

    # Print the progress to console
    print("{} - {}/{} - {:.2%}".format(output_dataset_file, i, total_count, i/total_count))

    # Increment the file counter
    i = i + 1
