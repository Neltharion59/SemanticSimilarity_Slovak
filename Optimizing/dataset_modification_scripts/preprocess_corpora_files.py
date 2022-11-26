# Runnable script preprocessing corpora, creating new versions of corpora (does not overwrite original corpora)
import os
from os import listdir
from os.path import isfile, join
import re

# Let's prepare the paths to read from and write to.
input_path = "./../resources/corpora/raw/"
output_path = "./../resources/corpora/"

# Let's prepare regexes to be used throughout this script.
corpus_input_file_name_pattern = re.compile(".*_sk.txt")

# Let's prepare list of corpus files to be preprocessed.
input_corpus_files = [x for x in listdir(input_path) if
                      isfile(join(input_path, x)) and corpus_input_file_name_pattern.match(x)]

# Let's loop over all input corpus files.
for input_corpus_file in input_corpus_files:
    # Prepare full path to both input and output file
    output_corpus_file = output_path + input_corpus_file
    input_corpus_file = input_path + input_corpus_file
    print(input_corpus_file)

    # Let's see how many lines of the corpus are already preprocessed
    try:
        with open(output_corpus_file, "r", encoding='utf-8') as output_file:
            processed_line_count = len(output_file.readlines())
    except FileNotFoundError:
        processed_line_count = 0

    current_line_counter = 0
    # Let's loop over all lines of the corpus
    with open(input_corpus_file, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            current_line_counter = current_line_counter + 1

            if current_line_counter <= processed_line_count:
                continue

            if input_corpus_file != './../resources/corpora/raw/2018_wiki_sk.txt':
                line = ' '.join(line.split('\t')[1:])

            words = line.split(' ')
            new_words = []
            for i in range(len(words)):
                word = words[i]
                word = word.replace('\n', '')

                while len(word) > 0 and not word[0].isalpha():
                    word = word[1:]
                while len(word) > 0 and not word[-1].isalpha():
                    word = word[:-1]

                if len(word) > 0:
                    new_words.append(word)

            line = ' '.join(new_words) + '\n'

            # Let's write the new line into output file
            with open(output_corpus_file, "a+", encoding='utf-8') as output_file:
                output_file.write(line)

                if current_line_counter % 100 == 0:
                    output_file.flush()
                    os.fsync(output_file)
