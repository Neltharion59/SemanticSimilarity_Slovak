# Library-like script providing knowledge-based similarity methods

from functools import reduce
import operator as op
from nltk.corpus import wordnet as wn

args_knowledge = {
    "synset_choice": ['1st_synsets', 'all_synsets'],
    "weighting": ['distance', 'ordinary'],
    "aggregation_operation": ['avg', 'max'],
    "wordnet": ['slk']
}

# Defines maximum possible value of Leacock-Chodorow similarity
leacock_chodorow_similarity_cap = 3.6375861597263857


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


# Calculates similarity of given pair of texts represented by strings
# using knowledge-based method specified by 'synset_sim_function' param.
# Provides convenient unified processing regardless of what method is used.
# Params: str, str, func<synset, synset> -> float, dict<str, str>
# Return: float
def calculate_knowledge_similarity_sentence(sentence1, sentence2, similarity_func_synset, args):
    for arg in args:
        if arg not in args_knowledge:
            raise ValueError('Knowledge-based method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_knowledge[arg]:
            raise ValueError('Knowledge-based method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    # Let's turn text into list of words
    words1 = sentence1.split(' ')
    words2 = sentence2.split(' ')

    # Prepare list of synsets for each word
    synsets1 = [wn.synsets(word, lang=args['wordnet']) for word in words1]
    synsets2 = [wn.synsets(word, lang=args['wordnet']) for word in words2]

    # If we only accept first syset of each word, let's get rid of other synsets
    if args['synset_choice'] == '1st_synsets':
        synsets1 = [x[:1] for x in synsets1]
        synsets2 = [x[:1] for x in synsets2]

    values = []
    # For each possible pair of synset list
    for i in range(len(synsets1)):
        for j in range(i, len(synsets2)):
            # Calculate weight of
            weight = 1/(1 + abs(i - j)) if args['weighting'] == 'distance' else 1
            # Calculate similarity of each pair of synsets and filter our 'None' values
            similarities = [similarity_func_synset(synsets1[i][k], synsets2[j][l]) for k in range(len(synsets1[i])) for l in range(k, len(synsets2[j]))]
            similarities = [x for x in similarities if x is not None]

            # If we have any similarity, appends to value list
            for similarity in similarities:
                values.append({
                    'weight': weight,
                    'similarity': similarity
                })

    result = None
    if len(values) == 0:
        result = 0
    elif args['aggregation_operation'] == 'avg':
        result = average([x['similarity'] for x in values], [x['weight'] for x in values])
    elif args['aggregation_operation'] == 'max':
        result = max([x['similarity'] for x in values])

    return result


# Calculates Wu-Palmer similarity of given pair of texts represented by strings
# Params: str, str, dict<str, str>
# Return: float
def wu_palmer_similarity_sentence(sentence1, sentence2, args):
    # Call the uniformly processing function to calculate the value
    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, lambda syn1, syn2: syn1.wup_similarity(syn2), args
    )

    return result


# Calculates path similarity of given pair of texts represented by strings
# Params: str, str, dict<str, str>
# Return: float
def path_similarity_sentence(sentence1, sentence2, args):
    # Call the uniformly processing function to calculate the value
    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, lambda syn1, syn2: syn1.path_similarity(syn2), args
    )

    return result


# Calculates Leacock-Chodorow similarity of given pair of texts represented by strings
# Params: str, str, dict<str, str>
# Return: float
def leacock_chodorow_similarity_sentence(sentence1, sentence2, args):
    # Call the uniformly processing function to calculate the value
    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, lambda syn1, syn2: None if syn1.pos() != syn2.pos() else (min(1, none_2_zero(syn1.lch_similarity(syn2))/leacock_chodorow_similarity_cap)), args
    )

    return result
