import json
from shutil import copyfile

paths = {
    '1st': './../resources/models_1/model_{}.{}',
    '2nd': './../resources/models/model_{}.{}'
}

model_ids = {
    '1st': [157, 149, 142, 86, 143, 151, 144, 88, 145, 121, 162, 92, 163, 171, 164, 172],
    '2nd': [276, 268, 245, 205, 246, 206, 263, 255, 264, 208, 281, 289, 266, 274, 283, 291]
}

datasets = [
    'semeval2012',
    'semeval2013',
    'semeval2014',
    'semeval2015',
    'semeval2016',
    'semevalall',
    'sick',
    'all'
]

target_directory = "./../resources/simplified_model_configs/"

for optimizer_run in model_ids:
    i = 0
    for model_id in model_ids[optimizer_run]:
        with open(paths[optimizer_run].format(model_id, 'json'), 'r') as file:
            config = json.loads(file.read())

        del config['vector']
        del config['fitness']
        del config['validation']
        del config['hyperparams']
        for input in config['inputs']:
            if 'corpus' in input['args']:
                del input['args']['corpus']

        dataset_version = 'raw' if i % 2 == 0 else 'lemma'
        dataset = datasets[(i // 2) % 8]

        new_file_name = target_directory + optimizer_run + '_' + dataset_version + '_' + dataset
        with open(new_file_name + '.json', 'w+') as file:
            file.write(json.dumps(config))

        copyfile(paths[optimizer_run].format(model_id, 'jolib'),  new_file_name + '.jolib')

        i += 1
