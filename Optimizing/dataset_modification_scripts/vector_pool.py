# Library-like script providing pool of wrappers of available alredy constructed vectors

from dataset_modification_scripts.vector_wrapper import VectorSpace

# In the end, we only had one construction method - very computationally expensive
available_construction_methods = ['hal']
available_corpora_names = ['2016_newscrawl_sk.txt']#, '2016_web_sk.txt', '2016_wikipedia_sk.txt', '2018_wiki_sk.txt', '2020_news_sk.txt']
vector_pool = {}
for construction_method in available_construction_methods:
    vector_pool[construction_method] = {}
    for corpus_name in available_corpora_names:
        # Raw version
        vector_pool[construction_method][corpus_name] = VectorSpace(construction_method, corpus_name)
        #corpus_name = corpus_name.replace('_sk.txt', '_sk_lemma.txt')
        # Lematized version
        #vector_pool[construction_method][corpus_name] = VectorSpace(construction_method, corpus_name)
