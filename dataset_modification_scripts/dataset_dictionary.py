# Runnable script lemmatizing datasets, creating new versions of datasets (does not overwrite original datasets)

from os import listdir
from os.path import isfile, join
import re
import json

from dataset_modification_scripts.dataset_pool import dataset_pool

word_pool_total = {}
for key in dataset_pool:

    word_pool_total[key] = {}
    word_pool = word_pool_total[key]

    for dataset in dataset_pool[key]:

        texts1, texts2 = dataset.load_dataset()

        for text in texts1:
            text = text.replace('\n', '').lower()
            words = text.split(' ')
            for word in words:
                if word not in word_pool:
                    word_pool[word] = 0
                word_pool[word] = word_pool[word] + 1

        for text in texts2:
            text = text.replace('\n', '').lower()
            words = text.split(' ')
            for word in words:
                if word not in word_pool:
                    word_pool[word] = 0
                word_pool[word] = word_pool[word] + 1
