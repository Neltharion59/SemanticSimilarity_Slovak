# Library-like script providing character-based similarity methods

from functools import reduce
from math import sqrt
import operator as op
import textdistance
from shared.custom_util import split_to_words


# Calculate Hamming similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float/None
def hamming(text1, text2, args):
    # Make sure text1 is the longer one (or same-sized one)
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    # Create match indicating vector
    mask = [1 if text1[i] == text2[i] else 0 for i in range(len(text2))]
    # Count matching positions
    matching_count = reduce(op.add, mask)
    # Apply normalization strategy
    if args["normalization_strategy"] == "longer":
        value = round(matching_count/len(text1), 2)
    elif args["normalization_strategy"] == "shorter":
        value = round(matching_count/len(text2), 2)
    elif args["normalization_strategy"] == "both":
        value = round(2 * matching_count/(len(text1) + len(text2)), 2)
    else:
        value = None

    return value


# Calculate levenshtein similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def levenshtein(text1, text2, args):
    return textdistance.levenshtein.normalized_similarity(text1, text2)


# Calculate damerau_levenshtein similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def damerau_levenshtein(text1, text2, args):
    return textdistance.damerau_levenshtein.normalized_similarity(text1, text2)


# Calculate jaro similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def jaro(text1, text2, args):
    return textdistance.jaro.normalized_similarity(text1, text2)


# Calculate jaro_winkler similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def jaro_winkler(text1, text2, args):
    return textdistance.jaro_winkler.normalized_similarity(text1, text2)


# Calculate needleman_wunsch similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def needleman_wunsch(text1, text2, args):
    return textdistance.needleman_wunsch.normalized_similarity(text1, text2)


# Calculate smith_waterman similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def smith_waterman(text1, text2, args):
    return textdistance.smith_waterman.normalized_similarity(text1, text2)

# Standard possibilities of args for term-based methods
args_set_based = {
    'level': ['character', 'text']
}


# Calculate jaccard similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def jaccard(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('Jaccard method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('Jaccard method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    if args['level'] == 'character':
        result = textdistance.jaccard.normalized_similarity(text1, text2)
    elif args['level'] == 'text':
        words1 = set(split_to_words(text1))
        words2 = set(split_to_words(text2))
        word_intersection = words1.intersection(words2)
        result = len(word_intersection)/(len(words1) + len(words2) - len(word_intersection))

    return result


# Calculate sorensen_dice similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def sorensen_dice(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('sorensen_dice method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('sorensen_dice method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    if args['level'] == 'character':
        result = textdistance.sorensen_dice.normalized_similarity(text1, text2)
    elif args['level'] == 'text':
        words1 = set(split_to_words(text1))
        words2 = set(split_to_words(text2))
        word_intersection = words1.intersection(words2)
        result = 2 * len(word_intersection)/(len(words1) + len(words2))

    return result


# Calculate overlap similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def overlap(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('overlap method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('overlap method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    if args['level'] == 'character':
        result = textdistance.overlap.normalized_similarity(text1, text2)
    elif args['level'] == 'text':
        words1 = set(split_to_words(text1))
        words2 = set(split_to_words(text2))
        word_intersection = words1.intersection(words2)
        result = len(word_intersection)/min(len(words1), len(words2))

    return result


# Calculate cosine similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def cosine(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('cosine method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('cosine method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    result = textdistance.cosine.normalized_similarity(text1, text2)

    return result


# Calculate lcsseq similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def lcsseq(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('lcsseq method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('lcsseq method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    result = textdistance.lcsseq.normalized_similarity(text1, text2)

    return result


# Calculate lcsstr similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def lcsstr(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('lcsstr method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('lcsstr method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    result = textdistance.lcsstr.normalized_similarity(text1, text2)

    return result


# Calculate ochiai similarity on given pair of strings using given parameters.
# Params: str, str, dict<string, string>
# Return: float
def ochiai(text1, text2, args):
    for arg in args:
        if arg not in args_set_based:
            raise ValueError('ochiai method received unknown argument - \'{}\''.format(arg))
        if args[arg] not in args_set_based[arg]:
            raise ValueError('ochiai method argument \'{}\' has unknown value - \'{}\''.format(arg, args[arg]))

    if args['level'] == 'character':
        set1 = set(text1)
        set2 = set(text2)
    elif args['level'] == 'text':
        set1 = set(split_to_words(text1))
        set2 = set(split_to_words(text2))
    set_intersection = set1.intersection(set2)
    result = len(set_intersection)/sqrt(len(set1) * len(set2))

    return result
