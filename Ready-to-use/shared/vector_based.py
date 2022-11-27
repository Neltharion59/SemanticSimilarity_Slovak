# Library-like script providing vector-based similarity methods (along with function to turn text to vector)

from decimal import Decimal
from math import sqrt
from operator import add
from scipy.spatial.distance import cosine as cos
from shared.custom_util import split_to_words

from shared.corpora_pool import corpora_pool
from shared.vector_pool import vector_pool

# Arg possibilities for vector-based methods
args_vector_based = {
    'vector_length': ['200', '400', '600', 'full'],
    'window_size': ['9'],
    'construction_method':  ['hal'],
    'vector_merge_strategy': ['add', 'add_pos_weight', 'add_power11_weight']
}
# Additional arg possibilities for 'p' parameter in minkowski distance
args_minkowski_p = [3, 4, 5]


# For given strings of text creates vector representation based on args.
# Params: str, str, dict<str, str>
# Return: list<float>, list<float>
def vectorize_text(text1, text2, args):
    # Load object with persisted vector representations of known words
    vector_object = vector_pool['vectors'].access_vector_object()

    words1 = split_to_words(text1)
    words2 = split_to_words(text2)

    vector_length = int(args['vector_length']) if args['vector_length'] != 'full' else len(next(iter(vector_object['vectors'][args['window_size']]['full'].values())))

    vector1 = [0] * vector_length
    vector2 = [0] * vector_length

    if args['vector_merge_strategy'] == 'add':
        for word in words1:
            if word in vector_object['vectors'][args['window_size']]['full']:
                vector1 = list(map(add, vector1, vector_object['vectors'][args['window_size']][args['vector_length']][word]))
        for word in words2:
            if word in vector_object['vectors'][args['window_size']]['full']:
                vector2 = list(map(add, vector2, vector_object['vectors'][args['window_size']][args['vector_length']][word]))

    elif args['vector_merge_strategy'] == 'add_pos_weight':
        for i in range(len(words1)):
            if words1[i] in vector_object['vectors'][args['window_size']]['full']:
                vector1 = list(map(lambda x, y: x + y * (i + 1), vector1, vector_object['vectors'][args['window_size']][args['vector_length']][words1[i]]))
        for i in range(len(words2)):
            if words2[i] in vector_object['vectors'][args['window_size']]['full']:
                vector2 = list(map(lambda x, y: x + y * (i + 1), vector2, vector_object['vectors'][args['window_size']][args['vector_length']][words2[i]]))

    elif args['vector_merge_strategy'] == 'add_power11_weight':
        for i in range(len(words1)):
            if words1[i] in vector_object['vectors'][args['window_size']]['full']:
                vector1 = list(map(lambda x, y: x + y * (11 ** i), vector1, vector_object['vectors'][args['window_size']][args['vector_length']][words1[i]]))
        for i in range(len(words2)):
            if words2[i] in vector_object['vectors'][args['window_size']]['full']:
                vector2 = list(map(lambda x, y: x + y * (11 ** i), vector2, vector_object['vectors'][args['window_size']][args['vector_length']][words2[i]]))

    else:
        raise ValueError('Unknown vector_merge_strategy: {}'.format(args['vector_merge_strategy']))

    return vector1, vector2


# For given pair of strings, calculates manhattan similarity using vector representation determined by args.
# Params: str, str, dict<str, str>
# Return: float
def manhattan(text1, text2, args):
    vector1, vector2 = vectorize_text(text1, text2, args)

    vector1_len = sum(map(abs, vector1))
    vector1_len = 1 if vector1_len == 0 else vector1_len
    vector1_normalized = list(map(lambda x: x / vector1_len, vector1))

    vector2_len = sum(map(abs, vector2))
    vector2_len = 1 if vector2_len == 0 else vector2_len
    vector2_normalized = list(map(lambda x: x / vector2_len, vector2))

    distance_vector_size = sum([abs(x - y) for x, y in zip(vector1_normalized, vector2_normalized)])
    similarity = 0 if distance_vector_size == 0 else 1 - distance_vector_size / 2

    return similarity


# For given pair of strings, calculates euclidean similarity using vector representation determined by args.
# Params: str, str, dict<str, str>
# Return: float
def euclidean(text1, text2, args):
    vector1, vector2 = vectorize_text(text1, text2, args)

    vector1_len = sqrt(sum(map(lambda x: x ** 2, vector1)))
    vector1_len = 1 if vector1_len == 0 else vector1_len
    vector1_normalized = list(map(lambda x: x/vector1_len, vector1))

    vector2_len = sqrt(sum(map(lambda x: x ** 2, vector2)))
    vector2_len = 1 if vector2_len == 0 else vector2_len
    vector2_normalized = list(map(lambda x: x / vector2_len, vector2))

    distance_vector_size = sqrt(sum([(x - y) ** 2 for x, y in zip(vector1_normalized, vector2_normalized)]))
    similarity = 0 if distance_vector_size == 0 else 1 - distance_vector_size / 2

    return similarity


# For given pair of strings, calculates minkowski similarity using vector representation determined by args.
# Params: str, str, dict<str, str>
# Return: float
def minkowski(text1, text2, args):
    vector1, vector2 = vectorize_text(text1, text2, args)

    temp_sum = Decimal(0)
    for number in vector1:
        temp_sum = Decimal(temp_sum + Decimal(Decimal(abs(number)) ** Decimal(args['p'])))
    vector1_len = Decimal(temp_sum ** Decimal(1 / args['p']))
    vector1_len = float(vector1_len)
    vector1_len = 1.0 if vector1_len == 0.0 else vector1_len
    vector1_normalized = list(map(lambda x: x/vector1_len, vector1))

    temp_sum = Decimal(0)
    for number in vector2:
        temp_sum = Decimal(temp_sum + Decimal(Decimal(abs(number)) ** Decimal(args['p'])))
    vector2_len = Decimal(temp_sum ** Decimal(1 / args['p']))
    vector2_len = float(vector2_len)
    vector2_len = 1.0 if vector2_len == 0.0 else vector2_len
    vector2_normalized = list(map(lambda x: x / vector2_len, vector2))

    distance_vector_size = (sum([round(abs(x - y) ** args['p'], ndigits=5) for x, y in zip(vector1_normalized, vector2_normalized)]) ** (1 / args['p']))
    similarity = 0 if distance_vector_size == 0 else 1 - distance_vector_size / 2

    return similarity


# For given pair of strings, calculates cosine similarity using vector representation determined by args.
# Params: str, str, dict<str, str>
# Return: float
def cosine_vector(text1, text2, args):
    vector1, vector2 = vectorize_text(text1, text2, args)

    similarity = 1 - cos(vector1, vector2)
    return similarity
