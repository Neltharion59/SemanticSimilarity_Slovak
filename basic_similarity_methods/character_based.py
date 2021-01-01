# Library-like script providing character-based similarity methods

from functools import reduce
import operator as op


# Calculate Hamming similarity on given pair of strings using given parameters.
# Params: str, str, dict
# Return: float/None
def hamming(word1, word2, args):
    # Make sure word1 is the longer one (or same-sized one)
    if len(word1) < len(word2):
        word1, word2 = word2, word1
    # Create match indicating vector
    mask = [1 if word1[i] == word2[i] else 0 for i in range(len(word2))]
    # Count matching positions
    matching_count = reduce(op.add, mask)
    # Apply normalization strategy
    if args["normalization_strategy"] == "longer":
        value = round(matching_count/len(word1), 2)
    elif args["normalization_strategy"] == "shorter":
        value = round(matching_count/len(word2), 2)
    elif args["normalization_strategy"] == "both":
        value = round(2 * matching_count/(len(word1) + len(word2)), 2)
    else:
        value = None

    return value
