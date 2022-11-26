from os import listdir
from os.path import isfile, join
import re
from nltk.corpus import wordnet as wn
from util.math import average
from dataset_modification_scripts.dataset_pool import dataset_pool
from dataset_modification_scripts.vector_pool import vector_pool


all_hits = {
    'raw': {},
    'lemma': {}
}


vectors = vector_pool['hal']['2016_newscrawl_sk.txt']
vectors.load_vector_object()


def count_hits(words):
    return [1 if word in vectors.vector_object['vectors']['9']['200'] else 0 for word in words]


print("Vectors loaded")
# Loop over all dataset versions
for dataset_version in dataset_pool:
    print(dataset_version)
    # Let's loop over all input dataset files
    for dataset in dataset_pool[dataset_version]:
        print(dataset.name)

        words1, words2 = dataset.load_dataset()

        sentences = [sen.split(' ') for sen in words1 + words2]

        this_hits = [count_hits(sentence) for sentence in sentences]

        words = []
        for sentence in sentences:
            words = words + sentence

        this_words_hits = sum(count_hits(words))

        all_hits[dataset_version][dataset.name] = {
            'avg_hit_rate_per_sentence': average([sum(hits) / len(hits) for hits in this_hits]),
            'unique_hits': this_words_hits/len(words)
        }


vectors.unload_vector_object()


for dataset_version in ['raw', 'lemma']:
    print(dataset_version)

    for dataset_name in all_hits[dataset_version]:
        print('\t' + dataset_name)
        print('\t\tavg_hit_rate_per_sentence: ' + str(round(all_hits[dataset_version][dataset_name]['avg_hit_rate_per_sentence'], 2)))
        print('\t\tunique_hits: ' + str(round(all_hits[dataset_version][dataset_name]['unique_hits'], 2)))