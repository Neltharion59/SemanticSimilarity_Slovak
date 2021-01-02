# Library-like script providing pool of all simple STS methods

from basic_similarity_methods.character_based import *
from basic_similarity_methods.knowledge_based import wu_palmer_similarity_sentence, path_similarity_sentence,\
    leacock_chodorow_similarity_sentence
from model_management.sts_method_wrappers import STSMethod


# Generates all combinations of parameters.
# Params: dict, dict
# Return: dict...
def all_arg_variations(arg_full_dict, output_arg_dict):

    # If we reached last param, let's yield the result
    if not bool(arg_full_dict):
        yield output_arg_dict

    # Loop over all params
    for parameter in arg_full_dict:
        # Loop over all values of current param
        for value in arg_full_dict[parameter]:
            # Add current value to current combination
            output_arg_dict_new = dict(output_arg_dict)
            output_arg_dict_new[parameter] = value

            # Jump to next param
            yield from all_arg_variations({i: arg_full_dict[i] for i in arg_full_dict if i != parameter}, output_arg_dict_new)
        break


# Adds
# Params: str, dict, func<... -> float>, dict<str, STSMethod>
# Return:
def add_to_method_pool(method_name, method_arg_possibilites, method_function, method_pool):
    for arg_variation in list(all_arg_variations(method_arg_possibilites, {})):
        sts_method = STSMethod(method_name, method_function, arg_variation)
        method_pool[sts_method.name] = sts_method


# Dict with all available simple STS methods
sts_method_pool = {}
# -----------------------------------------------------------------------------
# Add Hamming similarity
name = "hamming"
args = {
    "normalization_strategy": ["longer", "shorter", "both"]
}
add_to_method_pool(name, args, hamming, sts_method_pool)
# -----------------------------------------------------------------------------
# Add Wu-Palmer similarity
name = "wu_palmer"
args = {
    "sentence_merge_strategy": ['all_to_all', 'match_cutoff'],
    "synset_strategy": ['all_synsets', 'first_synsets', 'first', 'max', 'average'],
    "wordnet": ['slk']
}
add_to_method_pool(name, args, wu_palmer_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
# Add Path similarity
name = "path"
add_to_method_pool(name, args, path_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
# Add Leacock-Chodorow similarity
name = "leacock_chodorow"
add_to_method_pool(name, args, leacock_chodorow_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
