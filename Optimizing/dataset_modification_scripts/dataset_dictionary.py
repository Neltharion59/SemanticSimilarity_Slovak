# Runnable script creating dictionary of all words across all datasets.

from dataset_modification_scripts.dataset_pool import dataset_pool

word_pool_total = {}
# For each dataset version (raw vs. lemma).
for key in dataset_pool:

    word_pool_total[key] = {}
    word_pool = word_pool_total[key]
    # For each dataset.
    for dataset in dataset_pool[key]:
        # Load the persisted dataset.
        texts1, texts2 = dataset.load_dataset()
        # Extract all words from first texts of pairs.
        for text in texts1:
            text = text.replace('\n', '').lower()
            words = text.split(' ')
            for word in words:
                if word not in word_pool:
                    word_pool[word] = 0
                word_pool[word] = word_pool[word] + 1
        # Extract all words from second texts of pairs.
        for text in texts2:
            text = text.replace('\n', '').lower()
            words = text.split(' ')
            for word in words:
                if word not in word_pool:
                    word_pool[word] = 0
                word_pool[word] = word_pool[word] + 1
