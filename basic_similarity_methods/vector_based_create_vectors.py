import json
import os
from datetime import datetime, timedelta
from functools import reduce
import operator as op

from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd as scipy_svd

from dataset_modification_scripts.dataset_dictionary import word_pool_total as word_pool
from dataset_modification_scripts.corpora_pool import corpora_pool


# Handy function to easily process missing values
# Params: Any
# Return: Any
def none_2_zero(value):
    return 0 if value is None else value


# Handy function to calculate average of list of values (can handle empty list)
# Params: list<float>, [list<float>]
# Return: float
def average(values, weights=None):
    if len(values) == 0:
        return 0

    if weights is None:
        weights = [1] * len(values)

    if len(values) != len(weights):
        raise ValueError('Values and weights do not have the same length')

    return reduce(op.add, [value * weight for value, weight in zip(values, weights)]) / len(values)


with open("./../resources/stop_words.txt", 'r', encoding='utf-8') as file:
    stop_words = file.readline().replace(' ', '').split(',')


def is_stop_word(word):
    return word in stop_words


def is_word_vectorable(word, corpora_type):
    result = word in word_pool[corpora_type] and word.isalpha()
    return result


def is_word_featurable(word, corpora_type):
    result = is_word_vectorable(word, corpora_type)# or (len(word) > 2 and word.isalpha() and not is_stop_word(word))
    return result


# https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
def svd(matrix, n_elements):
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
        new_matrix[word] = vector

    return new_matrix


def corpus_window_slide(corpus, process_window_function, args):
    save_rate = 1000

    for window_size in args['window_size']:
        if window_size % 2 == 0:
            raise ValueError('Tried to create vectors with odd window size - {}'.format(window_size))
    max_window_size = max(args['window_size'])

    progress_track_file_name = "./../resources/temp/vector_building.txt"
    word_buffer = []

    try:
        with open(progress_track_file_name, 'r', encoding='utf-8') as progress_file:
            json_string = progress_file.read()
        progress_object = json.loads(json_string)
        first_line = False
    except FileNotFoundError:
        progress_object = {
            'already_processed_lines': 0,
            'already_processed_corpora': [],
            'word_position_dict': {},
            'matrix': {}
        }
        first_line = True

    if corpus.name in progress_object['already_processed_corpora']:
        return None, None

    times = []
    start_time = datetime.now()
    line_counter = 0
    for line in corpus.lines():
        line_counter = line_counter + 1

        if line_counter < progress_object['already_processed_lines']:
            continue
        elif line_counter == progress_object['already_processed_lines']:
            print('Starting at: {}'.format(line_counter + 1))
            continue

        line = line.replace('\n', '')
        word_buffer = word_buffer + line.split(' ')

        if line_counter % save_rate == 0:
            current_time = datetime.now()
            times.append((current_time - start_time).seconds)
            start_time = current_time

            print(
                '{}: line {}/{}. Time estimate: {}. One {}-piece batch takes: {}.'.format(
                    corpus.name,
                    line_counter,
                    corpus.size,
                    timedelta(seconds=((corpus.size - line_counter) / save_rate) * average(times)),
                    save_rate,
                    timedelta(seconds=average(times))
                )
            )
            progress_object['already_processed_lines'] = line_counter
            json_object = json.dumps(progress_object)
            with open(progress_track_file_name, 'w+', encoding='utf-8') as progress_file:
                progress_file.write(json_object)
                progress_file.flush()
                os.fsync(progress_file)

        if len(word_buffer) < max_window_size:
            continue

        if first_line:
            for i in range(max_window_size // 2):
                left = word_buffer[0:i]
                center = word_buffer[i]
                right = word_buffer[i + 1: i + 1 + max_window_size // 2]

                process_window_function(left, center, right, progress_object['matrix'], progress_object['word_position_dict'])

            first_line = False

        while len(word_buffer) >= max_window_size:
            left = word_buffer[0:max_window_size// 2]
            center = word_buffer[max_window_size // 2]
            right = word_buffer[max_window_size // 2 + 1:max_window_size]
            word_buffer.pop(0)

            process_window_function(left, center, right, progress_object['matrix'], progress_object['word_position_dict'])

    for i in range(len(word_buffer)):
        left = word_buffer[max(0, i - max_window_size // 2):i]
        center = word_buffer[i]
        right = word_buffer[i+1:] if i < len(word_buffer) - 1 else []

        process_window_function(left, center, right, progress_object['matrix'], progress_object['word_position_dict'])

    progress_object['already_processed_corpora'].append(corpus.name)
    progress_object['already_processed_lines'] = 0

    m = progress_object['matrix']
    wpd = progress_object['word_position_dict']

    progress_object['word_position_dict'] = {}
    progress_object['matrix'] = {}

    json_object = json.dumps(progress_object)
    with open(progress_track_file_name, 'w+', encoding='utf-8') as progress_file:
        progress_file.write(json_object)
        progress_file.flush()
        os.fsync(progress_file)

    return m, wpd


def create_vectors_hal(corpus, args):
    def process_window(left, center, right, matrix, word_position_dict):
        if is_word_vectorable(center, 'lemma' if corpus.is_lemma else 'raw'):
            if center not in matrix:
                matrix[center] = []
                if len(matrix.keys()) > 0:
                    for j in range(len(matrix[list(matrix.keys())[0]])):
                        matrix[center].append([])

            for j in reversed(range(len(left))):
                if is_word_featurable(left[j], 'lemma' if corpus.is_lemma else 'raw') and len(matrix.keys()) > 0:
                    if left[j] not in word_position_dict:
                        word_position_dict[left[j]] = len(matrix[list(matrix.keys())[0]])
                        for key in matrix:
                            matrix[key].append([])

                    matrix[center][word_position_dict[left[j]]].append(j + 1)

            for j in range(len(right)):
                if is_word_featurable(right[j], 'lemma' if corpus.is_lemma else 'raw') and len(matrix.keys()) > 0:
                    if right[j] not in word_position_dict:
                        word_position_dict[right[j]] = len(matrix[list(matrix.keys())[0]])
                        for key in matrix:
                            matrix[key].append([])

                    matrix[center][word_position_dict[right[j]]].append(j + 1)

        #print('Matrix {}x{}'.format(len(list(matrix.keys())), 0 if len(list(matrix.keys())) == 0 else len(matrix[list(matrix.keys())[0]])))

    matrix, word_position_dict = corpus_window_slide(corpus, process_window, args)

    if matrix is not None and word_position_dict is not None:
        matrices = {}
        for window_size in args['window_size']:
            matrices[window_size] = {}
            for word in matrix:
                matrices[window_size][word] = [reduce_matrix_cell([y for y in x if y < window_size // 2 + 1], window_size) for x in matrix[word]]

        for window_size in matrices:
            for word in matrices[window_size]:
                matrices[window_size][word] = matrices[window_size][word] + [matrices[window_size][word2][word_position_dict[word]] for word2 in matrix]

        result_matrices = {}
        for window_size in matrices:
            result_matrices[window_size] = {
                'full': matrices[window_size]
            }

            for vector_size in args['vector_size']:
                result_matrices[window_size][vector_size] = svd(matrices[window_size], vector_size)

        words = [''] * len(word_position_dict)
        for word in word_position_dict:
            words[word_position_dict[word]] = word

        return result_matrices, words

    else:
        return None, None


def reduce_matrix_cell(x, window_size):
    return average(x) / (len(x) * (window_size // 2)) if len(x) > 0 else 0
