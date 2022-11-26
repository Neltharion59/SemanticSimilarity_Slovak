# Runnable script calculating statistic values to be printed to console in CSV format.
# Provides overview of dataset sizes.

# Loop over all datasets, determine their properties and print them.
from dataset_modification_scripts.dataset_pool import dataset_pool

# For each dataset (properties are same for both versions, so no need to do that duplicitly).
for dataset in dataset_pool['raw']:
    words, _ = dataset.load_dataset()
    print("\t".join([dataset.name, str(len(words))]))
