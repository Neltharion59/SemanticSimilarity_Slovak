# Library-like script providing pool of all simple STS methods

from basic_similarity_methods.character_based import *
from basic_similarity_methods.knowledge_based import wu_palmer_similarity_sentence, path_similarity_sentence, \
    leacock_chodorow_similarity_sentence, args_knowledge
from basic_similarity_methods.vector_based import args_vector_based, args_minkowski_p, manhattan, euclidean, minkowski, \
    cosine_vector
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


# Adds method to method pool
# Params: str, dict, func<... -> float>, dict<str, STSMethod>
# Return:
def add_to_method_pool(method_name, method_arg_possibilites, method_function, method_pool):
    if len(list(all_arg_variations(method_arg_possibilites, {}))) == 1:
        sts_method = STSMethod(method_name, method_function, {})
        if method_name not in method_pool:
            method_pool[method_name] = []
        method_pool[method_name].append(sts_method)
    else:
        for arg_variation in list(all_arg_variations(method_arg_possibilites, {})):
            sts_method = STSMethod(method_name, method_function, arg_variation)
            if method_name not in method_pool:
                method_pool[method_name] = []
            method_pool[method_name].append(sts_method)


# Dict with all available simple STS methods
sts_method_pool = {}
# -----------------------------------------------------------------------------
# ------------------------   CHARACTER   --------------------------------------
# -----------------------------------------------------------------------------
# Add Hamming similarity
name = "hamming"
args_hamming = {
    "normalization_strategy": ["longer", "shorter", "both"]
}
add_to_method_pool(name, args_hamming, hamming, sts_method_pool)
# -----------------------------------------------------------------------------
# Add MLIPNS similarity
name = "mlipns"
args_mlipns = {
}
add_to_method_pool(name, args_mlipns, mlipns, sts_method_pool)
# -----------------------------------------------------------------------------
# Add levenshtein similarity
name = "levenshtein"
args_levenshtein = {
}
add_to_method_pool(name, args_levenshtein, levenshtein, sts_method_pool)
# -----------------------------------------------------------------------------
# Add damerau_levenshtein similarity
name = "damerau_levenshtein"
args_damerau_levenshtein = {
}
add_to_method_pool(name, args_damerau_levenshtein, damerau_levenshtein, sts_method_pool)
# -----------------------------------------------------------------------------
# Add jaro similarity
name = "jaro"
args_jaro = {
}
add_to_method_pool(name, args_jaro, jaro, sts_method_pool)
# -----------------------------------------------------------------------------
# Add jaro_winkler similarity
name = "jaro_winkler"
args_jaro_winkler = {
}
add_to_method_pool(name, args_jaro_winkler, jaro_winkler, sts_method_pool)
# -----------------------------------------------------------------------------
# Add needleman_wunsch similarity
name = "needleman_wunsch"
args_needleman_wunsch = {
}
add_to_method_pool(name, args_needleman_wunsch, needleman_wunsch, sts_method_pool)
# -----------------------------------------------------------------------------
# Add smith_waterman similarity
name = "smith_waterman"
args_smith_waterman = {
}
add_to_method_pool(name, args_smith_waterman, smith_waterman, sts_method_pool)
# -----------------------------------------------------------------------------
# Add jaccard similarity
name = "jaccard"
add_to_method_pool(name, args_set_based, jaccard, sts_method_pool)
# -----------------------------------------------------------------------------
# Add sorensen_dice similarity
name = "sorensen_dice"
add_to_method_pool(name, args_set_based, sorensen_dice, sts_method_pool)
# -----------------------------------------------------------------------------
# Add overlap similarity
name = "overlap"
add_to_method_pool(name, args_set_based, overlap, sts_method_pool)
# -----------------------------------------------------------------------------
# Add cosine similarity
name = "cosine"
add_to_method_pool(name, {}, cosine, sts_method_pool)
# -----------------------------------------------------------------------------
# Add lcsseq similarity
name = "lcsseq"
add_to_method_pool(name, {}, lcsseq, sts_method_pool)
# -----------------------------------------------------------------------------
# Add lcsstr similarity
name = "lcsstr"
add_to_method_pool(name, {}, lcsstr, sts_method_pool)
# -----------------------------------------------------------------------------
# Add ochiai similarity
name = "ochiai"
add_to_method_pool(name, args_set_based, ochiai, sts_method_pool)
# -----------------------------------------------------------------------------
# ---------------------------   VECTOR   --------------------------------------
# -----------------------------------------------------------------------------
# Add manhattan similarity
name = "manhattan"
add_to_method_pool(name, args_vector_based, manhattan, sts_method_pool)
# -----------------------------------------------------------------------------
# Add euclidean similarity
name = "euclidean"
add_to_method_pool(name, args_vector_based, euclidean, sts_method_pool)
# -----------------------------------------------------------------------------
# Add minkowski similarity
name = "minkowski"
args_minkowski = {key: value for (key, value) in args_vector_based.items()}
args_minkowski['p'] = args_minkowski_p
add_to_method_pool(name, args_minkowski, minkowski, sts_method_pool)
# -----------------------------------------------------------------------------
# Add cosine_vector similarity
name = "cosine_vector"
add_to_method_pool(name, args_vector_based, cosine_vector, sts_method_pool)
# -----------------------------------------------------------------------------
# ------------------------   KNOWLEDGE   --------------------------------------
# -----------------------------------------------------------------------------
# Add Wu-Palmer similarity
name = "wu_palmer"
add_to_method_pool(name, args_knowledge, wu_palmer_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
# Add Path similarity
name = "path"
add_to_method_pool(name, args_knowledge, path_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
# Add Leacock-Chodorow similarity
name = "leacock_chodorow"
add_to_method_pool(name, args_knowledge, leacock_chodorow_similarity_sentence, sts_method_pool)
# -----------------------------------------------------------------------------
