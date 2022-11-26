# Runnable script checking if each dataset has the same amount of records for each STS method
# it has calculated values of

from dataset_modification_scripts.dataset_pool import dataset_pool

# For each dataset version (raw vs. lemma).
for key in dataset_pool:
    # For each dataset.
    for dataset in dataset_pool[key]:
        # Load persisted dataset to check record count.
        words, _ = dataset.load_dataset()
        dataset_size = len(words)

        print('-' * 40 + '\nType: {} | Dataset: {} | Size: {} | Status:'.format(key, dataset.name, dataset_size))

        # Now load persisted STS values for this dataset.
        values = dataset.load_values()
        if len(list(values.keys())) == 0:
            print('Empty. SKIPPING')
            continue
        # Perform the record count check.
        faulty_configs = []
        for method_name in values:
            for config in values[method_name]:
                if len(config['values']) != dataset_size:
                    faulty_configs.append({
                        'method': method_name,
                        'args': config['args'],
                        'size': len(config['values'])
                    })

        # Finally, if there was anything wrong, tell the user.
        if len(faulty_configs) == 0:
            print('\t--->\tOK\t<---')
        else:
            print('\t--->\tBAD\t<---')
            for conf in faulty_configs:
                print('\tSize: {} | Method: {} | Args: {}'.format(conf['size'], conf['method'], conf['args']))


