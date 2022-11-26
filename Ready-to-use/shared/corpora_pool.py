# Library-like script providing pool of wrappers of available corpora

from shared.corpus_wrapper import Corpus

corpora_pool = {
    'raw': [
        Corpus('2016_newscrawl_sk.txt'),
        Corpus('2016_web_sk.txt'),
        Corpus('2016_wikipedia_sk.txt'),
        Corpus('2018_wiki_sk.txt'),
        Corpus('2020_news_sk.txt'),
    ]
}
corpora_pool['lemma'] = [Corpus(corpus.name.replace('_sk.txt', '_sk_lemma.txt')) for corpus in corpora_pool['raw']]
