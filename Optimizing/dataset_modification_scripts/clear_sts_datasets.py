# Runnable script modifying datasets to custom unified format

from os import listdir, walk
from os.path import isfile, join
import os
import errno
import re

# Prepare paths to directiores to work with
input_path = "./../resources/datasets/semeval-sts/"
output_path = "./../resources/datasets/sts_processed/"

# Prepare regexes to be used throughout this script
year_pattern = re.compile("\./\.\./resources/datasets/semeval\-sts/201[2-6]")
dataset_file_name_pattern = re.compile(".*\.tsv")
dataset_record_pattern = re.compile("^([0-5](\.[0-9]+)?)?\t.+\t.+")
dataset_record_explicit_sts_pattern = re.compile("^[0-5](\.[0-9]+)?")
dataset_record_explicit_sts_pattern_space_started = re.compile("^[0-5](\.[0-9]+)? ")

# LET'S TRANSFORM SEMEVAL DATASETS FIRST

# Get all subdirectories with SemEval datasets
year_dirs = [x[0] for x in walk(input_path) if year_pattern.match(x[0])]

# Loop over the subdirectiories with SemEval datasets (one subdirectory is for one year)
for year_dir in year_dirs:
    # Retrieve the year from directory name (its suffix)
    year = year_dir[-4:]

    # Let's get list of all dataset files in given year's directory
    dataset_files = [x for x in listdir(year_dir) if isfile(join(year_dir, x)) and dataset_file_name_pattern.match(x)]

    # Let's loop over all dataset files - read them and transform them
    for dataset_file in dataset_files:
        # Let's prepare full path to the dataset file
        file_name_input = year_dir + "/" + dataset_file

        # Initialize the resulting text (resulting transformed form of the dataset)
        result_text = ""

        # Sometimes if multiple same values follow each other, dataset only contains one of those values,
        # other lines are empty. So we need to be able to deal with that
        implicit_sts_front = []

        # Let's read from dataset file
        with open(file_name_input, "r", encoding='utf-8') as file_input:
            # Let's loop over lines of the dataset file
            for line in file_input:
                line = line.replace("\n", "")
                # If we have space as separator of sts and rest of record, fix it to tab
                if dataset_record_explicit_sts_pattern_space_started.match(line):
                    line = line.replace(" ", "\t", 1)
                # If this is not valid record, let's skip it
                if not dataset_record_pattern.match(line):
                    print("not valid record")
                    continue
                # Now we know we have tab-separated record with explicit sts or tab-prepended record with implicit sts
                # If it is tab-separated record with explicit sts
                if dataset_record_explicit_sts_pattern.match(line):
                    sts = dataset_record_explicit_sts_pattern.match(line).group(0)
                    for implicit_sts_record in implicit_sts_front:
                        result_text += sts + implicit_sts_record + "\n"
                    implicit_sts_front.clear()
                # If it is tab-prepended record with implicit sts
                else:
                    implicit_sts_front.append(line)
                    continue

                result_text += line + "\n"

        # Now we have transformed the dataset to pretty string. Let's write it to file then
        output_file_name = output_path + "dataset_sts_" + year + "_" + file_name_input.split('/')[-1].split('.')[0] + "_en.txt"

        # If the output path does not exist, let's create it
        # https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
        if not os.path.exists(os.path.dirname(output_file_name)):
            try:
                os.makedirs(os.path.dirname(output_file_name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Let's write the transformed datased to output file
        with open(output_file_name, "w+", encoding='utf-8') as output_file:
            output_file.write(result_text)

# NOW, TRANSFORM THE SICK DATASET

# Prepare directory path read from
input_path = "./../resources/datasets/sick2014/"

# Prepare new regexes to be used throughout this script
dataset_file_name_pattern = re.compile("SICK_.*\.txt")
dataset_record_pattern = re.compile("^[1-9][0-9]*\t.+\t+[0-5](\.[0-9]+)?\t.+$")

# Let's prepare list dataset files to read from
dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_file_name_pattern.match(x)]

# Initialize the resulting transformed dataset with empty string (dataset is split to multiple files)
result_text_full = ""

# Loop over dataset files
for dataset_file in dataset_files:
    # Initialize the current resulting transformed dataset with empty string
    result_text = ""
    input_file_name = input_path + dataset_file

    # Let's read from the current dataset file
    with open(input_file_name, 'r', encoding='utf-8') as input_file:
        # Let's loop over lines of the dataset
        for line in input_file:
            line = line.replace("\n", "")

            # Skip this line if it does not seem right
            if not dataset_record_pattern.match(line):
                continue

            # Let's modify the line to fit our format
            modified_line = "\t".join([line.split("\t")[x] for x in [3, 1, 2]]) + "\n"

            # Append the modified line to resulting text of current dataset and total dataset
            result_text_full += modified_line
            result_text += modified_line

    # Let's write the current transformed dataset to output file
    with open(output_path + "dataset_sick_" + input_file_name.split("/")[-1].split("_")[1] + "_en.txt", 'w+', encoding='utf-8') as output_file:
        output_file.write(result_text)

# Let's write the total transformed dataset to output file
with open(output_path + "dataset_sick_all_en.txt", 'w+', encoding='utf-8') as output_file:
    output_file.write(result_text_full)
