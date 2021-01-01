# Library-like script providing knowledge-based similarity methods

from functools import reduce
import operator as op
from nltk.corpus import wordnet as wn

# Defines maximum possible value of Leacock-Chodorow similarity
leacock_chodorow_similarity_cap = 3.6375861597263857


# Handy function to easily process missing values
# Params: object
# Return: float/int
def none_2_zero(value):
    return 0 if value is None else value


# Handy function to calculate average of list of values (can handle empty list)
# Params: list<float>
# Return: float/int
def average(values):
    return 0 if len(values) == 0 else reduce(op.add, values) / len(values)


# Handy function to retrieve list of synsets of given word in given WordNet
# Params: str, str
# Return: list<synset>
def prepare_synsets_single_word(word, wordnet):
    return wn.synsets(word, lang=wordnet)


# Handy function to prepare lists of synsets for both compared words
# Params: str, str, str
# Return: list<synset>, list<synset>
def prepare_synsets_two_words(word1, word2, wordnet):
    syn1 = prepare_synsets_single_word(word1, wordnet)
    syn2 = prepare_synsets_single_word(word2, wordnet)
    return syn1, syn2


# Calculates similarity of given pair of words represented by lists of synsets
# using knowledge-based method specified by 'synset_sim_function' param.
# Provides convenient unified processing regardless of what method is used.
# Params: list<synset>, list<synset>, str, func<synset,synset -> float>
# Return: float
def calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, synset_sim_function):
    result = None
    # Apply synset strategy - if 'first', only first synset of each word is considered
    if synset_strategy == 'first':
        synset1 = syn1[0]
        synset2 = syn2[0]
        # Make sure both synsets represent the same POS (e.g. nouns can be paired with nouns only)
        if synset1.pos() != synset2.pos():
            result = 0
        else:
            result = synset_sim_function(synset1, synset2)
        result = result if result is not None else 0
    # If we apply any other strategy than 'first', we will work with all the synsets
    else:
        # Calculate similarity each synset of 1st word with each synset of 2nd word
        values = []
        for synset1 in syn1:
            for synset2 in syn2:
                # Skip, if synsets are not of the same POS (e.g. nouns can be paired with nouns only)
                if synset1.pos() != synset2.pos():
                    continue
                similarity = synset_sim_function(synset1, synset2)
                # We're not interested in calculating with None (if no match between synsets)
                if similarity is not None:
                    values.append(similarity)
        # Make sure we don't have issues with empty list
        if len(values) == 0:
            result = 0
        elif synset_strategy == 'average':
            result = average(values)
        elif synset_strategy == 'max':
            result = max(values)
        # If synset strategy is anything else, this function should not have even been called, so do nothing

    return result


# Calculates Wu-Palmer similarity of given pair of words represented by strings
# Params: str, str, str, str
# Return: float
def wu_palmer_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    # Represent words as lists of synsets
    syn1, syn2 = prepare_synsets_two_words(word1, word2, wordnet)

    # If one of the words has no synsets, we cannot compute
    if len(syn1) == 0 or len(syn2) == 0:
        return None

    # Define similarity function of two synsets - Wu-Palmer
    def similarity_func(x, y):
        return x.wup_similarity(y)

    # Perform the calculation
    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)

    return result


# Calculates path similarity of given pair of words represented by strings
# Params: str, str, str, str
# Return: float
def path_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    # Represent words as lists of synsets
    syn1, syn2 = prepare_synsets_two_words(word1, word2, wordnet)

    # If one of the words has no synsets, we cannot compute
    if len(syn1) == 0 or len(syn2) == 0:
        return None

    # Define similarity function of two synsets - path
    def similarity_func(x, y):
        return x.path_similarity(y)

    # Perform the calculation
    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)

    return result


# Calculates Leacock-Chodorow similarity of given pair of words represented by strings
# Params: str, str, str, str
# Return: float
def leacock_chodorow_similarity_word(word1, word2, synset_strategy='first', wordnet='slk'):
    # Represent words as lists of synsets
    syn1, syn2 = prepare_synsets_two_words(word1, word2, wordnet)

    # If one of the words has no synsets, we cannot compute
    if len(syn1) == 0 or len(syn2) == 0:
        return None

    # Define similarity function of two synsets - path
    def similarity_func(x, y):
        return x.lch_similarity(y)

    # Perform the calculation
    result = calculate_knowledge_similarity_word(syn1, syn2, synset_strategy, similarity_func)

    # Normalize the value and clamp it (just in case)
    result = min(1, result/leacock_chodorow_similarity_cap)

    return result


# Calculates similarity of given pair of texts represented by strings
# using knowledge-based method specified by 'synset_sim_function' param.
# Provides convenient unified processing regardless of what method is used.
# Params: str, str, func, func, str, str, str
# Return: float
def calculate_knowledge_similarity_sentence(sentence1, sentence2, similarity_func_syn, similarity_func_word, sentence_merge_strategy, synset_strategy, wordnet):
    # Let's turn text into list of words
    words1 = sentence1.split(' ')
    words2 = sentence2.split(' ')

    # Apply synset strategy if needed - we will represent sentence as list of synsets
    if synset_strategy == 'all_synsets':
        # We'll concat all synsets of all words into single long list
        synsets1 = list(reduce(op.add, map(lambda x: prepare_synsets_single_word(x, wordnet), words1)))
        synsets2 = list(reduce(op.add, map(lambda x: prepare_synsets_single_word(x, wordnet), words2)))
    elif synset_strategy == 'first_synsets':
        # We'll replace each word for its first synset. If there is no synset for a word, the list shifts.
        synsets1 = list(reduce(op.add, map(lambda x: [] if len(x) == 0 else [x[0]], map(lambda x: prepare_synsets_single_word(x, wordnet), words1))))
        synsets2 = list(reduce(op.add, map(lambda x: [] if len(x) == 0 else [x[0]], map(lambda x: prepare_synsets_single_word(x, wordnet), words2))))

    # Let's prepare for the calculation. Regardless of strategy used, we will produce bunch of similarites
    # between words and average them in the end. So let's initialize array, to which we will collect those similarities.
    similarities = []

    # We would like to have uniform processing, whether we work with list of synsets or words.
    # So we based on strategy, we prepare list of words or synsets, let's call it collection of items.
    # We also need function that will calculate the similarity between two items.

    # If we want to work with list of synsets
    if synset_strategy in ['all_synsets', 'first_synsets']:
        # We have prepared lists of synsets in previous 'if' statement
        collection1, collection2 = synsets1, synsets2

        # Let's define similarity function between two synsets.
        # We received that as parameter, but we need some extra handling - appending to list of similarities.
        def sim_func(item1, item2, similarity_collector):
            # Only proceed if POS of synsets is same
            if item1.pos() == item2.pos():
                similarity = similarity_func_syn(item1, item2)
                # Only proceed if have an actual value
                if similarity is not None:
                     similarity_collector.append(similarity)
    # If we want to work with list of words
    elif synset_strategy in ['first', 'max', 'average']:
        # We have split the sentence to list of words in the beginning of this function
        collection1, collection2 = words1, words2

        # Let's define similarity function between two words.
        # We received that as parameter, but we need some extra handling - appending to list of similarities.
        def sim_func(item1, item2, similarity_collector):
            similarity_collector.append(similarity_func_word(item1, item2, synset_strategy, wordnet))

    # Let's apply sentence merge strategy on the collection of items,
    # i.e. how to create couples of items for which we will calculate similarities

    # Let's combine each with each
    if sentence_merge_strategy == 'all_to_all':
        for item1 in collection1:
            for item2 in collection2:
                sim_func(item1, item2, similarities)
    # Let's only combine same positions (0-0, 1-1, ...), and shorten the longer collection
    elif sentence_merge_strategy == 'match_cutoff':
        # Make the longer collection to be the same size as the shorter collection
        desired_length = min(len(collection1), len(collection2))
        collection1 = collection1[0:desired_length]
        collection2 = collection2[0:desired_length]

        # Let's calculate the similarities
        for i in range(desired_length):
            sim_func(collection1[i], collection2[i], similarities)

    # Let's filter out irrelevant values that are not really values
    similarities = list(filter(lambda x: x is not None, similarities))

    # Let's calculate the final result - as average of all values
    result = round(average(similarities), 2)

    return result


# Calculates Wu-Palmer similarity of given pair of texts represented by strings
# Params: str, str, dict
# Return: float
def wu_palmer_similarity_sentence(sentence1, sentence2, args):

    # Define the similarity function between synsets - to be used in the uniformly processing function
    def similarity_func_syn(synset1, synset2):
        return synset1.wup_similarity(synset2)

    # Prepare the similarity function between words - to be used in the uniformly processing function
    similarity_func_word = wu_palmer_similarity_word

    # Call the uniformly processing function to calculate the value
    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, similarity_func_syn, similarity_func_word,
        args['sentence_merge_strategy'], args['synset_strategy'], args['wordnet']
    )

    return result


# Calculates path similarity of given pair of texts represented by strings
# Params: str, str, dict
# Return: float
def path_similarity_sentence(sentence1, sentence2, args):

    # Define the similarity function between synsets - to be used in the uniformly processing function
    def similarity_func_syn(synset1, synset2):
        return synset1.path_similarity(synset2)

    # Prepare the similarity function between words - to be used in the uniformly processing function
    similarity_func_word = path_similarity_word

    # Call the uniformly processing function to calculate the value
    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, similarity_func_syn, similarity_func_word,
        args['sentence_merge_strategy'], args['synset_strategy'], args['wordnet']
    )

    return result


# Calculates Leacock-Chodorow similarity of given pair of texts represented by strings
# Params: str, str, dict
# Return: float
def leacock_chodorow_similarity_sentence(sentence1, sentence2, args):

    # Define the similarity function between synsets - to be used in the uniformly processing function
    def similarity_func_syn(synset1, synset2):
        return min(1, none_2_zero(synset1.lch_similarity(synset2))/leacock_chodorow_similarity_cap)

    # Prepare the similarity function between words - to be used in the uniformly processing function
    similarity_func_word = leacock_chodorow_similarity_word

    # Call the uniformly processing function to calculate the value
    result = calculate_knowledge_similarity_sentence(
        sentence1, sentence2, similarity_func_syn, similarity_func_word,
        args['sentence_merge_strategy'], args['synset_strategy'], args['wordnet']
    )

    return result
