# Runnable script lematizing corpora, creating new versions of corpora (does not overwrite original corpuss)
import os
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
        'text': text.lower(),
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
input_path = "./../resources/corpora/"
output_path = "./../resources/corpora/"

# Let's prepare regexes to be used throughout this script.
corpus_input_file_name_pattern = re.compile(".*_sk.txt")

# Let's prepare list of corpus files to be lemmatized.
input_corpus_files = [x for x in listdir(input_path) if isfile(join(input_path, x)) and corpus_input_file_name_pattern.match(x)]

# Let's loop over all input corpora files.
for input_corpus_file in input_corpus_files:
    # Prepare full path to both input and output file
    output_corpus_file = output_path + input_corpus_file.replace("_sk.txt", "_sk_lemma.txt")
    counter_file_path = output_path + input_corpus_file.replace("_sk.txt", "_sk_lemma_counter.txt")
    input_corpus_file = input_path + input_corpus_file

    print(input_corpus_file)

    # Let's see how many lines of the corpus are already lemmatized
    try:
        with open(counter_file_path, "r", encoding='utf-8') as counter_file:
            processed_line_count = int(counter_file.read())
    except FileNotFoundError:
        processed_line_count = 0

    print(processed_line_count)
    current_line_counter = 0
    batch = ""
    # Let's loop over all lines of the corpus
    with open(input_corpus_file, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            current_line_counter = current_line_counter + 1
            # If this line is already processed, let's not process it again
            if current_line_counter <= processed_line_count:
                continue
            # E-mail adresses mess with the lematizer tool we use.
            line = line.replace('@', ' ')
            # To maximize efficiency, we create batches as large as possible (limitation of the tool).
            if (len(batch) + len(line)) < 20000:
                batch = batch + ' ' + line
            else:
                # Perform the lematization.
                lemmatized_text = lemmatize(batch)
                words = lemmatized_text.split(' ')
                words = [word for word in words if word != '?']
                lemmatized_text = ' '.join(words) + '\n'
                batch = ""

                # Let's write the new line into output file
                with open(output_corpus_file, "a+", encoding='utf-8') as output_file:
                    output_file.write(lemmatized_text)
                    output_file.flush()
                    os.fsync(output_file)

                    # Update the progress in console
                    print("{}: Processing line {}".format(input_corpus_file, current_line_counter))

                    with open(counter_file_path, "w+", encoding='utf-8') as counter_file:
                        counter_file.write(str(current_line_counter))
                        counter_file.flush()
                        os.fsync(counter_file)

    # Once we have finished, there may be something left in the batch, so let's process it too.
    if batch != "":
        # Perform the lematization.
        lemmatized_text = lemmatize(batch)
        words = lemmatized_text.split(' ')
        words = [word for word in words if word != '?']
        lemmatized_text = ' '.join(words) + '\n'

        # Let's write the new line into output file
        with open(output_corpus_file, "a+", encoding='utf-8') as output_file:
            output_file.write(lemmatized_text)
            output_file.flush()
            os.fsync(output_file)
        with open(counter_file_path, "w+", encoding='utf-8') as counter_file:
            counter_file.write(str(current_line_counter))
            counter_file.flush()
            os.fsync(counter_file)
