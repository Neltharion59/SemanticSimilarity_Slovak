from basic_similarity_methods.character_based import *
from basic_similarity_methods.knowledge_based import wu_palmer_similarity_sentence, path_similarity_sentence,\
    leacock_chodorow_similarity_sentence
from model_management.sts_method_wrappers import STSMethod


def all_arg_variations(arg_full_dict, output_arg_dict):
    if not bool(arg_full_dict):
        yield output_arg_dict

    for parameter in arg_full_dict:
        for value in arg_full_dict[parameter]:
            output_arg_dict_new = dict(output_arg_dict)
            output_arg_dict_new[parameter] = value
            yield from all_arg_variations({i: arg_full_dict[i] for i in arg_full_dict if i != parameter}, output_arg_dict_new)
        break


def add_to_method_pool(method_name, method_arg_possibilites, method_function, method_pool):
    for arg_variation in list(all_arg_variations(method_arg_possibilites, {})):
        sts_method = STSMethod(method_name, method_function, arg_variation)
        method_pool[sts_method.name] = sts_method





sts_method_pool = {}
# -----------------------------------------------------------------------------
name = "hamming"
args = {
    "normalization_strategy": ["longer", "shorter", "both"]
}
add_to_method_pool(name, args, hamming, sts_method_pool)
# -----------------------------------------------------------------------------
name = "wu_palmer"
args = {
    "sentence_merge_strategy": ['all_to_all', 'match_cutoff'],
    "synset_strategy": ['all_synsets', 'first_synsets', 'first', 'max', 'average'],
    "wordnet": ['slk']
}
add_to_method_pool(name, args, wu_palmer_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
name = "path"
add_to_method_pool(name, args, path_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
name = "leacock_chodorow"
add_to_method_pool(name, args, leacock_chodorow_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------

