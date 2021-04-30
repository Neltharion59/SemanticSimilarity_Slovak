import json

from basic_similarity_methods.vector_based_create_vectors import create_vectors_hal
from dataset_modification_scripts.corpora_pool import corpora_pool

window_sizes = [11]
args = {
    'vector_size': [100, 200]
}
for key in corpora_pool:
    print('----------------------------')
    print(key)
    for corpus in corpora_pool[key]:
        print(corpus.name)

        vector_pool = {
            'hal': {}
        }

        for window_size in window_sizes:
            args['window_size'] = window_size
            vector_pool['hal'][window_size] = {}
            vector_pool['hal'][window_size]['vectors'], vector_pool['hal'][window_size]['words'] = create_vectors_hal(corpus, args)

        for key1 in vector_pool:
            print(key1)
            for key2 in vector_pool[key1]:
                print('\t' * 1 + str(key2))
                for key3 in vector_pool[key1][key2]:
                    print('\t' * 2 + str(key3))
                for key3 in vector_pool[key1][key2]['vectors']:
                    print('\t' * 3 + str(key3))
                    for key4 in vector_pool[key1][key2]['vectors'][key3]:
                        pass
                        #print('\t' * 3 + "{}: {}".format(str(key4), vector_pool[key1][key2]['vectors'][key3][key4]))
        exit()


