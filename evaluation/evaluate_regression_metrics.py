# Library-like script providing functions for evaluating method metrics

from scipy.stats.stats import pearsonr


def pearson(labels, predictions):
    return pearsonr(labels, predictions)[0]
