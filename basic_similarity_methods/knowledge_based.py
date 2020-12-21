from functools import reduce
import operator as op
from nltk.corpus import wordnet as wn
import nltk


def prepare_synsets(word1, word2, wordnet):
    syn1 = wn.synsets(word1, lang=wordnet)
    syn2 = wn.synsets(word2, lang=wordnet)
    return syn1, syn2


def calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, synset_sim_function):
    result = None
    if synset_strategy == 'first':
        synset1 = syn1[0]
        synset2 = syn2[0]
        if synset1.pos() != synset2.pos():
            result = 0
        else:
            result = synset_sim_function(synset1, synset2)
        result = result if result is not None else 0
    else:
        # Prepare similarity each with each
        values = []
        for synset1 in syn1:
            for synset2 in syn2:
                if synset1.pos() != synset2.pos():
                    continue
                similarity = synset_sim_function(synset1, synset2)
                if similarity is not None:
                    values.append(similarity)
        if len(values) == 0:
            result = 0
        elif synset_strategy == 'average':
            result = reduce(op.add, values) / len(values)
        elif synset_strategy == 'max':
            result = max(values)
    return result


def wu_palmer_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    syn1, syn2 = prepare_synsets(word1, word2, wordnet)

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    def similarity_func(x, y):
        return x.wup_similarity(y)

    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)
    return result


def path_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    syn1, syn2 = prepare_synsets(word1, word2, wordnet)

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    def similarity_func(x, y):
        return x.path_similarity(y)

    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)
    return result


def leacock_chodorow_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    syn1, syn2 = prepare_synsets(word1, word2, wordnet)
    cap = 3.6375861597263857

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    def similarity_func(x, y):
        return x.lch_similarity(y)

    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)
    result = min(1, result/cap)
    return result


# def wu_palmer_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
#     syn1, syn2 = prepare_synsets(word1, word2, wordnet)
#
#     if len(syn1) == 0 or len(syn2) == 0:
#         return None
#
#     result = None
#     if synset_strategy == 'first':
#         synset1 = syn1[0]
#         synset2 = syn2[0]
#         result = synset1.wup_similarity(synset2)
#         result = result if result is not None else 0
#     else:
#         # Prepare similarity each with each
#         values = []
#         for synset1 in syn1:
#             for synset2 in syn2:
#                 similarity = synset1.wup_similarity(synset2)
#                 if similarity is not None:
#                     values.append(similarity)
#         if len(values) == 0:
#             result = 0
#         elif synset_strategy == 'average':
#             result = reduce(op.add, values)/len(values)
#         elif synset_strategy == 'max':
#             result = max(values)
#     return result


words_pairs = [["pomaranč", "slabý"], ["šach", "hra"], ["pes", "pes"]]
similarities = [wu_palmer_similarity_word, path_similarity_word, leacock_chodorow_similarity_word]
synset_strategies = ['first', 'max', 'average']

for words_pair in words_pairs:
    for similarity in similarities:
        for synset_strategy in synset_strategies:
            print("{}-{} : {} - {}, {}".format(
                words_pair[0],
                words_pair[1],
                similarity(words_pair[0], words_pair[1], synset_strategy),
                similarity.__name__,
                synset_strategy
            ))
