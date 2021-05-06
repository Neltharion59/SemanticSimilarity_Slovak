# Runnable script calculating statistic values to be printed to console.
# Generate data in table form that can be imported to MS Excel so that it can be included in DP2 report.
# For each dataset, we have determine interesting properties.
# Row order: dataset_name, dataset size, STS value format (in Slovak)

# Loop over all datasets, determine their properties and print them
from dataset_modification_scripts.dataset_pool import dataset_pool

for dataset in dataset_pool['raw']:
    words, _ = dataset.load_dataset()
    print("\t".join([dataset.name, str(len(words))]))
