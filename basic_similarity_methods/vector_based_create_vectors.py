from functools import reduce
import operator as op

from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd as scipy_svd

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

    if args['window_size'] % 2 == 0:
        raise ValueError('Tried to create vectors with odd window size')

    word_buffer = []
    first_line = True

    line_counter = 0
    for line in corpus.lines():
        line_counter = line_counter + 1
        line = line.replace('\n', '')
        word_buffer = word_buffer + line.split(' ')

        if line_counter % 500 == 0:
            print('{}: line {}/{}'.format(corpus.name, line_counter, corpus.size))

        if len(word_buffer) < args['window_size']:
            continue

        if first_line:
            for i in range(args['window_size'] // 2):
                left = word_buffer[0:i]
                center = word_buffer[i]
                right = word_buffer[i + 1: i + 1 + args['window_size'] // 2]

                process_window_function(left, center, right)

            first_line = False

        while len(word_buffer) >= args['window_size']:
            left = word_buffer[0:args['window_size'] // 2]
            center = word_buffer[args['window_size'] // 2]
            right = word_buffer[args['window_size'] // 2 + 1:args['window_size']]
            word_buffer.pop(0)

            process_window_function(left, center, right)


    for i in range(len(word_buffer)):
        left = word_buffer[max(0, i - args['window_size'] // 2):i]
        center = word_buffer[i]
        right = word_buffer[i+1:] if i < len(word_buffer) - 1 else []

        process_window_function(left, center, right)


def create_vectors_hal(corpus, args):
    word_position_dict = {}
    matrix = {}

    def process_window(left, center, right):
        if center not in matrix:
            matrix[center] = []
            if len(matrix.keys()) > 0:
                for j in range(len(matrix[list(matrix.keys())[0]])):
                    matrix[center].append([])

        for j in range(len(left)):
            if left[j] not in word_position_dict:
                word_position_dict[left[j]] = len(matrix[list(matrix.keys())[0]])
                for key in matrix:
                    matrix[key].append([])

            matrix[center][word_position_dict[left[j]]].append(args['window_size'] // 2 - j)

        for j in range(len(right)):
            if right[j] not in word_position_dict:
                word_position_dict[right[j]] = len(matrix[list(matrix.keys())[0]])
                for key in matrix:
                    matrix[key].append([])

            matrix[center][word_position_dict[right[j]]].append(j + 1)

    corpus_window_slide(corpus, process_window, args)
    for word in matrix:
        matrix[word] = [average(x)/(len(x) * (args['window_size'] // 2)) if len(x) > 0 else 0 for x in matrix[word]]
        #matrix[word] = matrix[word] + [matrix[row_word][word_position_dict[word]] for row_word in matrix.keys()]

    print('New')
    print(len(matrix[list(matrix.keys())[0]]))
    for word1 in matrix:
            #print('-' * 20 + '\n{}: {}\n{}: {}'.format(word1, matrix[word1], word2, matrix[word2]))
        matrix[word1] = matrix[word1] + [matrix[word2][word_position_dict[word1]] for word2 in matrix]
    print(len(matrix[list(matrix.keys())[0]]))
    matrices = {
        'full': matrix
    }
    for vector_size in args['vector_size']:
        matrices[vector_size] = svd(matrix, vector_size)

    words = [''] * len(word_position_dict)
    for word in word_position_dict:
        words[word_position_dict[word]] = word

    return matrices, words
