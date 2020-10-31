from os import listdir, walk
from os.path import isfile, join
import os
import errno
import re

# SemEval datasets

input_path = "./../resources/datasets/semeval-sts/"
output_path = "./../resources/datasets/sts_processed/"

# Regexes
year_pattern = re.compile("\./\.\./resources/datasets/semeval\-sts/201[2-6]")
dataset_file_name_pattern = re.compile(".*\.tsv")
dataset_record_pattern = re.compile("^([0-5](\.[0-9]+)?)?\t.+\t.+")
dataset_record_explicit_sts_pattern = re.compile("^[0-5](\.[0-9]+)?")
dataset_record_explicit_sts_pattern_space_started = re.compile("^[0-5](\.[0-9]+)? ")

# Get all directories with SemEval datasets
year_dirs = [x[0] for x in walk(input_path) if year_pattern.match(x[0])]

for year_dir in year_dirs:
    year = year_dir[-4:]

    # let's get list of all dataset files in given year's directory
    dataset_files = [x for x in listdir(year_dir) if isfile(join(year_dir, x)) and dataset_file_name_pattern.match(x)]
    for dataset_file in dataset_files:
        file_name_input = year_dir + "/" + dataset_file
        result_text = ""
        implicit_sts_front = []

        with open(file_name_input, "r", encoding='utf-8') as file_input:
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

        output_file_name = output_path + "dataset_sts_" + year + "_" + file_name_input.split('/')[-1].split('.')[0] + "_en.txt"
        # https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
        if not os.path.exists(os.path.dirname(output_file_name)):
            try:
                os.makedirs(os.path.dirname(output_file_name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(output_file_name, "w+", encoding='utf-8') as output_file:
            output_file.write(result_text)

# SICK dataset

input_path = "./../resources/datasets/sick2014/"
dataset_file_name_pattern = re.compile("SICK_.*\.txt")
dataset_record_pattern = re.compile("^[1-9][0-9]*\t.+\t+[0-5](\.[0-9]+)?\t.+$")

dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_file_name_pattern.match(x)]

result_text_full = ""
for dataset_file in dataset_files:
    result_text = ""
    input_file_name = input_path + dataset_file
    with open(input_file_name, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.replace("\n", "")
            if not dataset_record_pattern.match(line):
                continue
            modified_line = "\t".join([line.split("\t")[x] for x in [3, 1, 2]]) + "\n"
            result_text_full += modified_line
            result_text += modified_line
    with open(output_path + "dataset_sick_" + input_file_name.split("/")[-1].split("_")[1] + "_en.txt", 'w+', encoding='utf-8') as output_file:
        output_file.write(result_text)

with open(output_path + "dataset_sick_all_en.txt", 'w+', encoding='utf-8') as output_file:
    output_file.write(result_text_full)