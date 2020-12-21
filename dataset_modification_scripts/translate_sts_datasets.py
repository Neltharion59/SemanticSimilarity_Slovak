from os import listdir
from os.path import isfile, join
import re
from time import sleep
from googletrans import Translator

translator = Translator()
slovak_letters = 'áäčďéíľĺňóôŕšťúýžÁČĎÉÍĽĹŇÓŠŤÚÝŽ'


def translate(input_text):
    translated_text = translator.translate(input_text, scr='en', dest='sk').text
    return translated_text


def is_slovak(input_text):
    for char in input_text:
        if char in slovak_letters:
            return True
    return False


input_path = "./../resources/datasets/sts_processed/"
output_path = "./../resources/datasets/sts_processed/"

dataset_input_file_name_pattern = re.compile(".*_en.txt")

input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

for input_dataset_file in input_dataset_files:
    output_dataset_file = output_path + input_dataset_file.replace("_en.txt", "_sk.txt")
    input_dataset_file = input_path + input_dataset_file

    existing_output_file_lines = []
    try:
        with open(output_dataset_file, 'r', encoding='utf-8') as output_file:
            existing_output_file_lines = output_file.readlines()
    except FileNotFoundError:
        pass

    input_file_lines = []
    with open(input_dataset_file, 'r', encoding='utf-8') as input_file:
        input_file_lines = input_file.readlines()

    total_count = len(input_file_lines)
    if not len(existing_output_file_lines) == len(input_file_lines):
        input_file_lines = input_file_lines[len(existing_output_file_lines):]
    else:
        continue

    i = len(existing_output_file_lines) + 1
    for line in input_file_lines:
        line = line.replace('\n', '')
        tokens = line.split('\t')

        tokens[1] = translate(tokens[1])
        tokens[2] = translate(tokens[2])

        output_text = "\t".join(tokens) + "\n"

        with open(output_dataset_file, "a+", encoding='utf-8') as output_file:
            output_file.write(output_text)

        print("{} - {}/{} - {:.2%}".format(output_dataset_file, i, total_count, i/total_count))
        i = i + 1

# Check if all translations worked

input_path = "./../resources/datasets/sts_processed/"

dataset_input_file_name_pattern = re.compile(".*_sk.txt")

input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

for input_dataset_file in input_dataset_files:
    input_dataset_file = input_path + input_dataset_file
    lines = []
    with open(input_dataset_file, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    for i in range(len(lines)):
        print("Checking {} - {}/{} - {:.2%}".format(input_dataset_file, i, len(lines), i/len(lines)))
        line = lines[i].replace("\n", "")
        tokens = line.split("\t")
        try:
            change = False
            if not is_slovak(tokens[1]):
                tokens[1] = translate(tokens[1])
                change = True
            if not is_slovak(tokens[2]):
                tokens[2] = translate(tokens[2])
                change = True
            if change:
                print("Translation needed \n{}\n{}".format(tokens[1], tokens[2]))
                lines[i] = "\t".join(tokens) + "\n"
                sleep(0.5)
            else:
                continue
        except AttributeError:
            pass

        if i % 50 == 0:
            with open(input_dataset_file, 'w+', encoding='utf-8') as output_file:
                output_file.writelines(lines)

    with open(input_dataset_file, 'w+', encoding='utf-8') as output_file:
        output_file.writelines(lines)


