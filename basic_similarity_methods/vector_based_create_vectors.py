# Library-like script providing functions for vector construction from corpora

import json
import os
from datetime import datetime, timedelta

from numpy import array, diag, zeros
from scipy.linalg import svd as scipy_svd

from dataset_modification_scripts.dataset_dictionary import word_pool_total as word_pool
from util.math import average

# Prepare list of stop words - obtained from http://text.fiit.stuba.sk/zoznam_stop_slov.php
with open("./../resources/stop_words.txt", 'r', encoding='utf-8') as file:
    stop_words = file.readline().replace(' ', '').split(',')


# Checks if given word is a stop word, using the loaded list.
# Params: str
# Return: bool
def is_stop_word(word):
    return word in stop_words


# Checks if we want to represent the given word as vector. It must:
#   - be in dictionary of words from datasets (we do not need to vectorize other words),
#   - only consist of letters (we do not need email adresses or other malformations),
#   - be longer than 2 letters (shorter words are either typos or appear in texts so often, they won't help us),
#   - not be a stop word (same case as 2-letter words).
# Params: str, 'raw' | 'lemma'
# Return: bool
def is_word_vectorable(word, corpora_type):
    result = word in word_pool[corpora_type] and word.isalpha() and len(word) > 2 and not is_stop_word(word)
    return result


# Checks if we want to use the word as column in vector representation.
# In the end, we decided this should be the same as vectorable condition.
# Keeping it as separate function lets us easily change this in the future.
# Params: str, 'raw' | 'lemma'
# Return: bool
def is_word_featurable(word, corpora_type):
    result = is_word_vectorable(word, corpora_type)
    return result


# Performs SVD on a maxtrix - reduces number of columns to desired amount with preserving structural relationships.
# Source: https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
# Params: list<list<float>>, int, [int]
# Return: list<list<float>>
def svd(matrix, n_elements, ndigits=2):
    A = array([matrix[key] for key in matrix])
    # Singular-value decomposition
    U, s, _ = scipy_svd(A)
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[0], :A.shape[0]] = diag(s)
    # select
    Sigma = Sigma[:, :n_elements]
    # transform
    T = U.dot(Sigma)
    T = T.tolist()

    new_matrix = {}
    for word, vector in zip(matrix, T):
        new_matrix[word] = [round(x, ndigits=ndigits) for x in vector]

    return new_matrix


# Slides through a given corpus and constructs full-size vectors representing words
# (does not apply SVD to make then shorter).
# Params: Corpus, func<list<str>, str, list<str>, dict<str, list<float>>, dict<str, int>> -> None, dict<str, str>
# Return: dict<str, list<float>>, dict<str, int>
def corpus_window_slide(corpus, process_window_function, args):
    if corpus.is_lemma:
        save_rate = 1
        line_limit = 50
    else:
        save_rate = 500
        line_limit = 8000

    # Windows need even size (x%2 == 1), so that they have center.
    for window_size in args['window_size']:
        if window_size % 2 == 0:
            raise ValueError('Tried to create vectors with odd window size - {}'.format(window_size))
    max_window_size = max(args['window_size'])

    progress_track_file_name = "./../resources/temp/vector_building.txt"
    word_buffer = []

    # Check if we haven't already started, but program got interrupted for some reason.
    try:
        # If so, let's load the progress and continue where we left.
        with open(progress_track_file_name, 'r', encoding='utf-8') as progress_file:
            json_string = progress_file.read()
        progress_object = json.loads(json_string)
        first_line = False
    except FileNotFoundError:
        # Otherwise we are starting anew, so we need to initialize the progress object.
        progress_object = {
            'already_processed_lines': 0,
            'already_processed_corpora': [],
            'word_position_dict': {},
            'matrix': {}
        }
        first_line = True

    # If this corpus has already been processed, let's finish, as we have nothing to do anymore.
    if corpus.name in progress_object['already_processed_corpora']:
        return None, None

    times = []
    start_time = datetime.now()
    line_counter = 0
    # Let's loop over lines of corpus to construct the vectors.
    for line in corpus.lines():
        line_counter = line_counter + 1

        # We may have started processing the corpus but did not finish,
        # so let's not process the lines that have already been processed.
        if line_counter < progress_object['already_processed_lines']:
            continue
        elif line_counter == progress_object['already_processed_lines']:
            print('Starting at: {}'.format(line_counter + 1))
            continue

        # Corpora are huge. Constructing vectors from tens of thousands of lines
        # inevitably crashes because of memory limits
        # (even if 20 000-line matrix can be constructed, SVD will crash it).
        if line_counter > line_limit:
            break

        line = line.replace('\n', '').lower()
        word_buffer = word_buffer + line.split(' ')

        # We cannot save progress after each line, as it would significantly slow down the process.
        # So we save progress every on every 'x' iterations.
        if line_counter % save_rate == 0:
            current_time = datetime.now()
            times.append((current_time - start_time).seconds)
            start_time = current_time

            # Let's notify of progress - staring at empty console for hours is bad.
            print(
                '{}: line {}/{}. Time estimate: {}. One {}-piece batch takes: {}.'.format(
                    corpus.name,
                    line_counter,
                    line_limit,
                    timedelta(seconds=((line_limit - line_counter) / save_rate) * average(times[-3 if len(times) >= 3 else 0:])),
                    save_rate,
                    timedelta(seconds=average(times[-3 if len(times) >= 3 else 0:]))
                )
            )

            # Update the progress object and persist it. The program may be interrupted, so let's force writing to disk.
            progress_object['already_processed_lines'] = line_counter
            json_object = json.dumps(progress_object)
            with open(progress_track_file_name, 'w+', encoding='utf-8') as progress_file:
                progress_file.write(json_object)
                progress_file.flush()
                os.fsync(progress_file)

        # If the line is too short for our window size, we need to continue to next line.
        if len(word_buffer) < max_window_size:
            continue

        # In case of first line, the window will be outside of bounds of text, we need to handle that separately.
        if first_line:
            for i in range(max_window_size // 2):
                # Prepare the words seen by window.
                left = word_buffer[0:i]
                center = word_buffer[i]
                right = word_buffer[i + 1: i + 1 + max_window_size // 2]
                # Process the words seen by window (append them to matrix).
                process_window_function(left, center, right, progress_object['matrix'], progress_object['word_position_dict'])

            first_line = False

        # While the available text is big enough (smaller than window size),
        # we may process it.
        while len(word_buffer) >= max_window_size:
            # Prepare the words seen by window.
            left = word_buffer[0:max_window_size // 2]
            center = word_buffer[max_window_size // 2]
            right = word_buffer[max_window_size // 2 + 1:max_window_size]
            word_buffer.pop(0)
            # Process the words seen by window (append them to matrix).
            process_window_function(left, center, right, progress_object['matrix'], progress_object['word_position_dict'])

    # After enough lines, we are finishing.
    # There are still some words in the buffer and we are in the same situation as with first line -
    # buffer is smaller than window size, so we need to process it with extra care.
    for i in range(len(word_buffer)):
        # Prepare the words seen by window.
        left = word_buffer[max(0, i - max_window_size // 2):i]
        center = word_buffer[i]
        right = word_buffer[i+1:] if i < len(word_buffer) - 1 else []
        # Process the words seen by window (append them to matrix).
        process_window_function(left, center, right, progress_object['matrix'], progress_object['word_position_dict'])

    # Update progress object.
    progress_object['already_processed_corpora'].append(corpus.name)
    progress_object['already_processed_lines'] = 0
    # Prepare the results that this function returns:
    #   - matrix of numerical representations of words,
    #   - position of each word as a column.
    m = progress_object['matrix']
    wpd = progress_object['word_position_dict']
    # Empty the progress object to be used on next run.
    progress_object['word_position_dict'] = {}
    progress_object['matrix'] = {}
    # Persist the progress object. We are paranoid, so let's force writing on disk.
    json_object = json.dumps(progress_object)
    with open(progress_track_file_name, 'w+', encoding='utf-8') as progress_file:
        progress_file.write(json_object)
        progress_file.flush()
        os.fsync(progress_file)

    return m, wpd


# Creates HAL method vectors for words in given corpus.
# Params: Corpus, dict<str, str>
# Return: dict<int, dict<int, dict<str, list<float>>>>, dict<str, int>
def create_vectors_hal(corpus, args):
    # Function (that does not need to exist in global context) that will process single window within text.
    # Params: list<str>, str, list<str>, dict<str, list<float>>, dict<str, int>
    # Return: None
    def process_window(left, center, right, matrix, word_position_dict):
        if is_word_vectorable(center, 'lemma' if corpus.is_lemma else 'raw'):
            # First make sure we have a record for the center word.
            if center not in matrix:
                matrix[center] = []
                if len(matrix.keys()) > 0:
                    for j in range(len(matrix[list(matrix.keys())[0]])):
                        matrix[center].append([])
            if center not in word_position_dict:
                word_position_dict[center] = len(matrix[list(matrix.keys())[0]])
                for key in matrix:
                    matrix[key].append([])

            # Process the left side of window.
            for j in reversed(range(len(left))):
                if is_word_featurable(left[j], 'lemma' if corpus.is_lemma else 'raw') and len(matrix.keys()) > 0:
                    if left[j] not in word_position_dict:
                        word_position_dict[left[j]] = len(matrix[list(matrix.keys())[0]])
                        for key in matrix:
                            matrix[key].append([])

                    matrix[center][word_position_dict[left[j]]].append(j + 1)

            # Process the right side of window.
            for j in range(len(right)):
                if is_word_featurable(right[j], 'lemma' if corpus.is_lemma else 'raw') and len(matrix.keys()) > 0:
                    if right[j] not in word_position_dict:
                        word_position_dict[right[j]] = len(matrix[list(matrix.keys())[0]])
                        for key in matrix:
                            matrix[key].append([])

                    matrix[center][word_position_dict[right[j]]].append(j + 1)

    # Perform the matrix construction
    matrix, word_position_dict = corpus_window_slide(corpus, process_window, args)

    print('Finished sliding')

    # If the matrix creation yielded anything (was performed), let's processed.
    if matrix is not None and word_position_dict is not None:
        print('Building matrices')
        # Create a new matrix for each defined window size.
        matrices = {}
        i = 1
        for window_size in args['window_size']:
            matrices[window_size] = {}
            print('Starting building matrix {}/{} - {}%'.format(i, len(args['window_size']), round(i/len(args['window_size']), ndigits=2) * 100))
            j = 1
            j_max = len(list(matrix.keys()))
            for word in matrix:
                print('Building matrix {}/{} - {}% ===> Word {}/{} - {}%'.format(
                    i, len(args['window_size']), round(i / len(args['window_size']), ndigits=2) * 100,
                    j, j_max, round(j / j_max, ndigits=2) * 100
                ))
                matrices[window_size][word] = [round(reduce_matrix_cell([y for y in x if y < window_size // 2 + 1], window_size), ndigits=2) for x in matrix[word]]
                j = j + 1
            i = i + 1
        # Concatenate row and column for each word - just as HAL method dictates it.
        print('Concating row and column')
        i = 1
        i_max = len(list(matrices.keys()))
        for window_size in matrices:
            print('Starting concatenating matrix {}/{} - {}%'.format(i, i_max, round(i / i_max, ndigits=2) * 100))
            j = 1
            j_max = len(list(matrices[window_size].keys()))
            for word in matrices[window_size]:
                print('Concatenating matrix {}/{} - {}% ===> Word \'{}\' {}/{} - {}%'.format(
                    i, i_max, round(i / i_max, ndigits=2) * 100,
                    word,
                    j, j_max, round(j / j_max, ndigits=2) * 100
                ))
                matrices[window_size][word] = matrices[window_size][word] + [matrices[window_size][word2][word_position_dict[word]] for word2 in matrix]
                j = j + 1
            i = i + 1
        # Perform SVD on each matrix to each defined vector size.
        print('SVDing matrices')
        result_matrices = {}
        i = 1
        i_max = len(list(matrices.keys()))
        for window_size in matrices:
            result_matrices[window_size] = {
                'full': matrices[window_size]
            }
            print('Starting SVDing matrix {}/{} - {}%'.format(i, i_max, round(i / i_max, ndigits=2) * 100))
            j = 1
            j_max = len(args['vector_size'])
            for vector_size in args['vector_size']:
                print('SVDing matrix {}/{} - {}% ===> Vector size {}/{} - {}%'.format(
                    i, i_max, round(i / i_max, ndigits=2) * 100,
                    j, j_max, round(j / j_max, ndigits=2) * 100
                ))
                result_matrices[window_size][vector_size] = svd(matrices[window_size], vector_size)
                j = j + 1
            i = i + 1
        # We also need to know on which position which word is used as column (feature)
        print('Creating word list')
        words = [''] * len(list(word_position_dict.keys()))
        i = 1
        i_max = len(list(word_position_dict.keys()))
        for word in word_position_dict:
            print('Appending to word list {}/{} - {}%'.format(i, i_max, round(i / i_max, ndigits=2) * 100))
            words[word_position_dict[word]] = word
            i = i + 1

        return result_matrices, words

    else:
        return None, None


# Reduces single matrix cell (list of floats) to single value (float).
# Params: list<float>, int
# Return: float
def reduce_matrix_cell(x, window_size):
    return average(x) / (len(x) * (window_size // 2)) if len(x) > 0 else 0
