from nltk.corpus import wordnet as wn


# Add params - synset choice strategy (first, perform average, ...)
def wu_palmer(word1, word2, wordnet='slk'):
    syn1 = wn.synsets(word1, lang=wordnet)
    syn2 = wn.synsets(word2, lang=wordnet)

    if len(syn1) == 0 or len(syn2) == 0:
        return None

    syn1 = syn1[0]
    syn2 = syn2[0]
    
    return wn.wup_similarity(syn1, syn2)


methods = [wu_palmer, wu_palmer]
results = [method("pekne", "d") for method in methods]
print(results)
