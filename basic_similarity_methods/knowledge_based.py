from nltk.corpus import wordnet as wn
import nltk


# Add params - synset choice strategy (first, perform average, ...)
def wu_palmer(word1, word2, wordnet='slk'):
    syn1 = wn.synsets(word1, lang=wordnet)
    syn2 = wn.synsets(word2, lang=wordnet)

    print(syn1)
    print(syn2)
    if len(syn1) == 0 or len(syn2) == 0:
        return None

    syn1 = syn1[0]
    syn2 = syn2[0]

    return syn1.wup_similarity(syn1, syn2)


print(wu_palmer("pes", "pekný"))
