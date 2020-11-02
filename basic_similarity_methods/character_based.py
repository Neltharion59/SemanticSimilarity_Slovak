# TODO parametrize normalized output scale? perhaps 0-5 should not be hard-coded
# TODO perhaps scale should be simply concern of output "layer"
from functools import reduce
import operator as op


def hamming(word1, word2, args):
    if len(word1) < len(word2):
        word1, word2 = word2, word1
    mask = [1 if word1[i] == word2[i] else 0 for i in range(len(word2))]
    matching_count = reduce(op.add, mask)
    if args["normalization_strategy"] == "longer":
        value = round(matching_count/len(word1), 2)
    elif args["normalization_strategy"] == "shorter":
        value = round(matching_count/len(word2), 2)
    return value
