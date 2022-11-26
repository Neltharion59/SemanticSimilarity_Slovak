# Runnable script calculating statistic values to be printed to console in CSV format.
# For each method category, get method and configuration count.

from model_management.sts_method_pool import sts_method_pool, string_based_name_list, corpus_based_name_list, \
    knowledge_based_name_list

counts = {
    'sb': {
        'method_count': 0,
        'configuration_count': 0
    },
    'cb': {
        'method_count': 0,
        'configuration_count': 0
    },
    'kb': {
        'method_count': 0,
        'configuration_count': 0
    }
}

for method_name in sts_method_pool:
    if method_name in string_based_name_list:
        key = 'sb'
    elif method_name in corpus_based_name_list:
        key = 'cb'
    elif method_name in knowledge_based_name_list:
        key = 'kb'
    else:
        raise ValueError('Method {} does not fit any category.'.format(method_name))

    counts[key]['method_count'] = counts[key]['method_count'] + 1
    counts[key]['configuration_count'] = counts[key]['configuration_count'] + len(sts_method_pool[method_name])

headers = ['Kategória', 'Počet metód', 'Počet konfigurácií']
category_mapping = {
    'sb': 'Založené na reťazcoch',
    'cb': 'Založené na korpusoch',
    'kb': 'Založené na znalostiach'
}

tabulable_values = [headers] + [[category_mapping[key], counts[key]['method_count'], counts[key]['configuration_count']] for key in counts]
tabulable_values.append(
    [
        'Spolu',
        sum([counts[key]['method_count'] for key in counts]),
        sum([counts[key]['configuration_count'] for key in counts])
    ]
)

for row in tabulable_values:
    print(','.join([str(x) for x in row]))
