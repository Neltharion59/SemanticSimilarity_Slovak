from basic_similarity_methods.character_based import *
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


sts_method_pool = {}

name = "hamming"
args = {
    "normalization_strategy": ["longer", "shorter", "both"]
}

for arg_variation in list(all_arg_variations(args, {})):
    sts_method = STSMethod(name, hamming, arg_variation)
    sts_method_pool[sts_method.name] = sts_method
