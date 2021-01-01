# Runnable script translating English datasets to Slovak

from os import listdir
from os.path import isfile, join
import re
from time import sleep
from googletrans import Translator


# Let's prepare the Translator object to be used to translate
translator = Translator()
# Letters, presence of which in string is approximation of Slovak language detection
slovak_letters = 'áäčďéíľĺňóôŕšťúýžÁČĎÉÍĽĹŇÓŠŤÚÝŽ'


# Translate given text using google translate form English to Slovak
# Params: str
# Return: str
def translate(input_text):
    translated_text = translator.translate(input_text, scr='en', dest='sk').text
    return translated_text


# Check whether given text is in Slovak, so it does not need translating.
# Approximation by checking for special chars, so sometimes we will try to translate already translated text
# Params: str
# Return: str
def is_slovak(input_text):
    for char in input_text:
        if char in slovak_letters:
            return True
    return False


# Let's prepare the paths to read from and write to.
input_path = "./../resources/datasets/sts_processed/"
output_path = "./../resources/datasets/sts_processed/"


# Let's prepare regexes to be used throughout this script.
dataset_input_file_name_pattern = re.compile(".*_en.txt")


# Let's prepare list of dataset files to be translated
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]


# Let's loop over all input dataset files
for input_dataset_file in input_dataset_files:
    # Prepare full name of output and input file
    output_dataset_file = output_path + input_dataset_file.replace("_en.txt", "_sk.txt")
    input_dataset_file = input_path + input_dataset_file

    # # Let's see which lines of the dataset are already translated
    existing_output_file_lines = []
    try:
        with open(output_dataset_file, 'r', encoding='utf-8') as output_file:
            existing_output_file_lines = output_file.readlines()
    except FileNotFoundError:
        pass

    # Let's prepare all lines of the dataset
    input_file_lines = []
    with open(input_dataset_file, 'r', encoding='utf-8') as input_file:
        input_file_lines = input_file.readlines()

    # Let's prepare only the lines that need to be translated
    total_count = len(input_file_lines)
    if not len(existing_output_file_lines) == len(input_file_lines):
        input_file_lines = input_file_lines[len(existing_output_file_lines):]
    else:
        continue

    # Initialize the line counter (from the last line that was lemmatized)
    i = len(existing_output_file_lines) + 1

    # Let's loop over lines of input file (only those that need to be lemmatized)
    for line in input_file_lines:
        line = line.replace('\n', '')

        # Let's split the line to tokens (similarity score, text1, text2)
        tokens = line.split('\t')

        # Translate both texts
        tokens[1] = translate(tokens[1])
        tokens[2] = translate(tokens[2])

        # Reconstruct the line with translated sentences
        output_text = "\t".join(tokens) + "\n"

        # Let's write the new line into output file
        with open(output_dataset_file, "a+", encoding='utf-8') as output_file:
            output_file.write(output_text)

        # Update the progress in console
        print("{} - {}/{} - {:.2%}".format(output_dataset_file, i, total_count, i/total_count))

        # Increment line counter
        i = i + 1

# Check if all translations worked
input_path = "./../resources/datasets/sts_processed/"

# Let's prepare regexes to be used throughout this script.
dataset_input_file_name_pattern = re.compile(".*_sk.txt")

# Let's prepare list of dataset files to be checked
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

# Let's loop over all input dataset files.
for input_dataset_file in input_dataset_files:
    # Prepare full path to both input and output file
    input_dataset_file = input_path + input_dataset_file

    # Let's read all the lines of current dataset file
    lines = []
    with open(input_dataset_file, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    # Let's loop over all the lines
    for i in range(len(lines)):
        # Print the current progress
        print("Checking {} - {}/{} - {:.2%}".format(input_dataset_file, i, len(lines), i/len(lines)))
        line = lines[i].replace("\n", "")

        # Let's split the line to tokens (similarity score, text1, text2)
        tokens = line.split("\t")
        try:
            change = False
            # Check if 1st text need to translated and do it if needed.
            if not is_slovak(tokens[1]):
                tokens[1] = translate(tokens[1])
                change = True
            # Check if 2nd text need to translated and do it if needed.
            if not is_slovak(tokens[2]):
                tokens[2] = translate(tokens[2])
                change = True
            # If translation was performed let's reconstruct the line
            # and overwrite it in transient representation of file.
            if change:
                # Print info that translation was performed
                print("Translation needed \n{}\n{}".format(tokens[1], tokens[2]))
                lines[i] = "\t".join(tokens) + "\n"
                sleep(0.5)
            else:
                continue
        except AttributeError:
            pass

        # Each 50 lines, let's write the modified file to disk
        if i % 50 == 0:
            with open(input_dataset_file, 'w+', encoding='utf-8') as output_file:
                output_file.writelines(lines)

    # In the end, let's write the modified file to disk
    with open(input_dataset_file, 'w+', encoding='utf-8') as output_file:
        output_file.writelines(lines)
