from functools import reduce
import operator as op


# Handy function to easily process missing values
# Params: Any
# Return: Any
def none_2_zero(value):
    return 0 if value is None else value


# Handy function to calculate average of list of values (can handle empty list)
# Params: list<float>, [list<float>]
# Return: float
def average(values, weights=None):
    if len(values) == 0:
        return 0

    if weights is None:
        weights = [1] * len(values)

    if len(values) != len(weights):
        raise ValueError('Values and weights do not have the same length')

    return reduce(op.add, [value * weight for value, weight in zip(values, weights)]) / len(values)