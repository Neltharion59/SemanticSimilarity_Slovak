from os import listdir
from os.path import isfile, join
import re
from nltk.corpus import wordnet as wn
from util.math import average
from dataset_modification_scripts.dataset_pool import dataset_pool


lang = 'slk'


all_hits = {
    'raw': {},
    'lemma': {}
}


def count_hits(words):
    return [1 if len(wn.synsets(word, lang=lang)) > 0 else 0 for word in words]


# Loop over all dataset versions
for dataset_version in dataset_pool:
    # Let's loop over all input dataset files
    for dataset in dataset_pool[dataset_version]:

        words1, words2 = dataset.load_dataset()

        sentences = [sen.split(' ') for sen in words1 + words2]

        this_hits = [count_hits(sentence) for sentence in sentences]

        words = []
        for sentence in sentences:
            words = words + sentence

        this_words_hits = sum([1 if len(wn.synsets(word, lang=lang)) > 0 else 0 for word in words])

        all_hits[dataset_version][dataset.name] = {
            'avg_hit_rate_per_sentence': average([sum(hits) / len(hits) for hits in this_hits]),
            'unique_hits': this_words_hits/len(words)
        }


for dataset_version in ['raw', 'lemma']:
    print(dataset_version)

    for dataset_name in all_hits[dataset_version]:
        print('\t' + dataset_name)
        print('\t\tavg_hit_rate_per_sentence: ' + str(round(all_hits[dataset_version][dataset_name]['avg_hit_rate_per_sentence'], 2)))
        print('\t\tunique_hits: ' + str(round(all_hits[dataset_version][dataset_name]['unique_hits'], 2)))