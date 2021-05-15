# Library-like script providing functions for evaluating method metrics. Currently not much here.

from scipy.stats.stats import pearsonr


# Wrapper for getting pearson value easily (scipy also provides p-value)
def pearson(labels, predictions):
    return pearsonr(labels, predictions)[0]
