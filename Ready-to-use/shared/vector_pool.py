# Library-like script providing pool of wrappers of available alredy constructed vectors

from shared.vector_wrapper import VectorSpace

available_corpora_names = ['vectors']
vector_pool = {}
for corpus_name in available_corpora_names:
    vector_pool[corpus_name] = VectorSpace(corpus_name)
