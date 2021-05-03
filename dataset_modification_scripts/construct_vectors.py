import json
import os

from basic_similarity_methods.vector_based_create_vectors import create_vectors_hal
from dataset_modification_scripts.corpora_pool import corpora_pool


args = {
    'vector_size': [200, 400, 600],
    'window_size': [17]
}
for key in corpora_pool:
    print('----------------------------')
    print(key)
    for corpus in corpora_pool[key]:
        print(corpus.name)
        vector_file_name = './../resources/vectors/hal/' + corpus.name + "_" + str(args['window_size'][0]) + '.txt'

        matrices, words = create_vectors_hal(corpus, args)

        if matrices is None and words is None:
            continue

        print('Creating vector pool')

        vector_pool = {
            'words': words,
            'vectors': matrices
        }

        print('Dumping to JSON')
        json_vectors = json.dumps(vector_pool)
        print('Writing to file')
        with open(vector_file_name, 'w+', encoding='utf-8') as vector_file:
            vector_file.write(json_vectors)
            vector_file.flush()
            os.fsync(vector_file)
        print('Written in file')
