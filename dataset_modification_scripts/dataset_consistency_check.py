from dataset_modification_scripts.dataset_pool import dataset_pool

for key in dataset_pool:
    for dataset in dataset_pool[key]:
        words, _ = dataset.load_dataset()
        dataset_size = len(words)

        print('-' * 40 + '\nType: {} | Dataset: {} | Size: {} | Status:'.format(key, dataset.name, dataset_size))

        values = dataset.load_values()
        if len(list(values.keys())) == 0:
            print('Empty. SKIPPING')
            continue

        faulty_configs = []
        for method_name in values:
            for config in values[method_name]:
                if len(config['values']) != dataset_size:
                    faulty_configs.append({
                        'method': method_name,
                        'args': config['args'],
                        'size': len(config['values'])
                    })

        if len(faulty_configs) == 0:
            print('\t--->\tOK\t<---')
        else:
            print('\t--->\tBAD\t<---')
            for conf in faulty_configs:
                print('\tSize: {} | Method: {} | Args: {}'.format(conf['size'], conf['method'], conf['args']))


