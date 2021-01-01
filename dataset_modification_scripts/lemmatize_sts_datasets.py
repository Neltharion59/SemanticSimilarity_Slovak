# Runnable script lemmatizing datasets, creating new versions of datasets (does not overwrite original datasets)

from os import listdir
from os.path import isfile, join
import re
import json
from time import sleep
from requests import post

# Define address of lemmatizer API
lemmatizer_address = 'http://arl6.library.sk/nlp4sk/api'


# Lemmatize given text using lemmatizer API
# Params: str
# Return: str
def lemmatize(text):
    # Need to sleep for a moment to make sure we do not put too much load on the lemmatizer server
    sleep(0.2)

    # Prepare the JSON to be sent within request
    request_json = {
        # Text to be lemmatized
        'text': text,
        # Security element. We used our own API key to have unlimited access,
        # but we do not want to publish it. This key is valid thought, but the access is limited.
        'apikey': 'DEMO',
        # Which tool we want to use
        'lemmatizer': 'DictionaryLemmatizer'
    }

    # Let's send the request to the server and receive the response.
    response = post(lemmatizer_address, allow_redirects=True, data=request_json)

    # Let's transform the response JSON to list of lemmas.
    result = list(
        filter(
            lambda x: x is not None,
            map(
                lambda x: x['lemma'][0] if 'lemma' in x and len(x['lemma']) > 0 else x['word'].lowercase(),
                json.loads(response.text)
            )
        )
    )

    # Let's turn the list of lemmas to space-separated string.
    result = ' '.join(result)

    return result


# Let's prepare the paths to read from and write to.
input_path = "./../resources/datasets/sts_processed/"
output_path = "./../resources/datasets/sts_processed/"

# Let's prepare regexes to be used throughout this script.
dataset_input_file_name_pattern = re.compile(".*_sk.txt")

# Let's prepare list of dataset files to be lemmatized.
input_dataset_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and dataset_input_file_name_pattern.match(x)]

# Let's loop over all input dataset files.
for input_dataset_file in input_dataset_files:
    # Prepare full path to both input and output file
    output_dataset_file = output_path + input_dataset_file.replace("_sk.txt", "_sk_lemma.txt")
    input_dataset_file = input_path + input_dataset_file

    # Let's see which lines of the dataset are already lemmatized
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

    # Let's prepare only the lines that need to be lemmatized
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
        sentence1, sentence2 = tokens[1], tokens[2]

        # Lemmatize both texts
        tokens[1] = lemmatize(tokens[1])
        tokens[2] = lemmatize(tokens[2])

        # Reconstruct the line with lemmatized sentences
        output_text = "\t".join(tokens) + "\n"

        # Let's write the new line into output file
        with open(output_dataset_file, "a+", encoding='utf-8') as output_file:
            output_file.write(output_text)

        # Update the progress in console
        print("{} - {}/{} - {:.2%}".format(output_dataset_file, i, total_count, i/total_count))

        # Increment line counter
        i = i + 1
