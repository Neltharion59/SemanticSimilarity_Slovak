# Runnable script that creates and persists vector representations for known words using corpora.

import json
import os

from basic_similarity_methods.vector_based_create_vectors import create_vectors_hal
from dataset_modification_scripts.corpora_pool import corpora_pool

# Arg possibilities of vector contruction. Were limited to be computable in reasonable time.
args = {
    'vector_size': [200, 400, 600],
    'window_size': [17]
}
# For each corpus version - raw vs. lemma
for key in corpora_pool:
    print('----------------------------')
    print(key)
    # For each available corpus
    for corpus in corpora_pool[key]:
        print(corpus.name)
        vector_file_name = './../resources/vectors/hal/' + corpus.name + "_" + str(args['window_size'][0]) + '.txt'

        # Construct vectors for given corpus
        matrices, words = create_vectors_hal(corpus, args)

        # If nothing was returned, it means vectors for this corpus have already been constructed,
        # so let's skip to another corpus.
        if matrices is None and words is None:
            continue

        print('Creating vector pool')

        # Prepare the vector object to be persisted.
        vector_pool = {
            'words': words,
            'vectors': matrices
        }

        # Write it to disk (forcefully, as we are paranoid).
        print('Dumping to JSON')
        json_vectors = json.dumps(vector_pool)
        print('Writing to file')
        with open(vector_file_name, 'w+', encoding='utf-8') as vector_file:
            vector_file.write(json_vectors)
            vector_file.flush()
            os.fsync(vector_file)
        print('Written in file')
