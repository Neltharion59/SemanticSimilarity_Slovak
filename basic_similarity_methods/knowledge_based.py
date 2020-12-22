from functools import reduce
import operator as op
from nltk.corpus import wordnet as wn
import nltk


leacock_chodorow_similarity_cap = 3.6375861597263857


def none_2_zero(value):
    return 0 if value is None else value


def average(values):
    return 0 if len(values) == 0 else reduce(op.add, values) / len(values)


def prepare_synsets_single_word(word, wordnet):
    return wn.synsets(word, lang=wordnet)


def prepare_synsets_two_words(word1, word2, wordnet):
    syn1 = prepare_synsets_single_word(word1, wordnet)
    syn2 = prepare_synsets_single_word(word2, wordnet)
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
            result = average(values)
        elif synset_strategy == 'max':
            result = max(values)
    return result


def wu_palmer_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    syn1, syn2 = prepare_synsets_two_words(word1, word2, wordnet)

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    def similarity_func(x, y):
        return x.wup_similarity(y)

    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)
    return result


def path_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    syn1, syn2 = prepare_synsets_two_words(word1, word2, wordnet)

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    def similarity_func(x, y):
        return x.path_similarity(y)

    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)
    return result


def leacock_chodorow_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    syn1, syn2 = prepare_synsets_two_words(word1, word2, wordnet)

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    def similarity_func(x, y):
        return x.lch_similarity(y)

    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)
    result = min(1, result/leacock_chodorow_similarity_cap)
    return result


def calculate_knowledge_similarity_sentence(sentence1, sentence2, similarity_func_syn, similarity_func_word, sentence_merge_strategy, synset_strategy, wordnet):
    words1 = sentence1.split(' ')
    words2 = sentence2.split(' ')

    if synset_strategy == 'all_synsets':
        synsets1 = list(reduce(op.add, map(lambda x: prepare_synsets_single_word(x, wordnet), words1)))
        synsets2 = list(reduce(op.add, map(lambda x: prepare_synsets_single_word(x, wordnet), words2)))
    elif synset_strategy == 'first_synsets':
        synsets1 = list(reduce(op.add, map(lambda x: [] if len(x) == 0 else [x[0]], map(lambda x: prepare_synsets_single_word(x, wordnet), words1))))
        synsets2 = list(reduce(op.add, map(lambda x: [] if len(x) == 0 else [x[0]], map(lambda x: prepare_synsets_single_word(x, wordnet), words2))))

    similarities = []
    if synset_strategy in ['all_synsets', 'first_synsets']:
        collection1, collection2 = synsets1, synsets2

        def sim_func(item1, item2, similarity_collector):
            if item1.pos() == item2.pos():
                similarity = similarity_func_syn(item1, item2)
                if similarity is not None:
                     similarity_collector.append(similarity)
    elif synset_strategy in ['first', 'max', 'average']:
        collection1, collection2 = words1, words2

        def sim_func(item1, item2, similarity_collector):
            similarity_collector.append(similarity_func_word(item1, item2, synset_strategy, wordnet))

    if sentence_merge_strategy == 'all_to_all':
        for item1 in collection1:
            for item2 in collection2:
                sim_func(item1, item2, similarities)
    elif sentence_merge_strategy == 'match_cutoff':
        desired_length = min(len(collection1), len(collection2))
        collection1 = collection1[0:desired_length]
        collection2 = collection2[0:desired_length]
        for i in range(desired_length):
            sim_func(collection1[i], collection2[i], similarities)

    result = round(average(similarities), 2)

    return result


def wu_palmer_similarity_sentence(sentence1, sentence2, args):
    def similarity_func_syn(synset1, synset2):
        return synset1.wup_similarity(synset2)
    similarity_func_word = wu_palmer_similarity_word

    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, similarity_func_syn, similarity_func_word,
        args['sentence_merge_strategy'], args['synset_strategy'], args['wordnet']
    )
    return result


def path_similarity_sentence(sentence1, sentence2, args):
    def similarity_func_syn(synset1, synset2):
        return synset1.path_similarity(synset2)
    similarity_func_word = path_similarity_word

    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, similarity_func_syn, similarity_func_word,
        args['sentence_merge_strategy'], args['synset_strategy'], args['wordnet']
    )
    return result


def leacock_chodorow_similarity_sentence(sentence1, sentence2, args):
    def similarity_func_syn(synset1, synset2):
        return min(1, none_2_zero(synset1.lch_similarity(synset2))/leacock_chodorow_similarity_cap)
    similarity_func_word = leacock_chodorow_similarity_word

    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, similarity_func_syn, similarity_func_word,
        args['sentence_merge_strategy'], args['synset_strategy'], args['wordnet']
    )
    return result

# for sentence_pair in sentence_pairs:
#     for similarity in similarities:
#         for sentence_merge_strategy in args['sentence_merge_strategy']:
#             for synset_strategy in args['synset_strategy']:
#                 curr_args = {
#                     'sentence_merge_strategy': sentence_merge_strategy,
#                     'synset_strategy': synset_strategy,
#                     'wordnet': 'slk'
#                 }
#                 print("{}-{} : {} - {}, {}, {}".format(
#                     sentence_pair[0],
#                     sentence_pair[1],
#                     similarity(sentence_pair[0], sentence_pair[1], curr_args),
#                     similarity.__name__,
#                     sentence_merge_strategy,
#                     synset_strategy
#                 ))
# exit()
