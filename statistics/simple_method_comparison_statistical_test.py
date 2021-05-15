# Runnable script calculating statistic values to be printed to console in R code format.
# Tests statistical significance of improvement achieved by lematization for each method type.

import math

from dataset_modification_scripts.dataset_pool import dataset_pool
from dataset_modification_scripts.dataset_wrapper import gold_standard_name
from evaluation.evaluate_regression_metrics import pearson

alpha = 0.05
p_value_digit_count = 6

values = {}

for key in dataset_pool:
    values[key] = {}

    for dataset in dataset_pool[key]:

        persisted_values = dataset.load_values()

        for method_name in persisted_values:

            if method_name == gold_standard_name:
                continue

            pearson_values = []
            for record in persisted_values[method_name]:
                pearson_value = pearson([x/5 for x in persisted_values['gold_standard'][0]['values']], record['values'])
                if not math.isnan(pearson_value):
                    pearson_values.append(pearson_value)

            aggregated_pearson = max(pearson_values)

            if method_name not in values[key]:
                values[key][method_name] = []

            values[key][method_name].append(aggregated_pearson)

print('setwd("C:/Users/lukas/Desktop/FIIT/DP/DP3/R")')
file_name = 'lema_vs_no_lemma_significance.txt'

print(
    'write(\n\tpaste(' +
    ','.join(['"Metóda"', '"Test"', '"p Hodnota"', '"Zlepšenie lematizáciou"', 'sep = ","'])
    + '\t),'
)
print('\tfile="./{}",'.format(file_name))
print('\tappend=TRUE')
print(')')

for method_name in values['raw'].keys():
    print('#' + '-' * 40)
    print('# ' + method_name)
    print('#' + '-' * 40 + '\n')

    for key in values:
        print('values_{}_{} <-'.format(method_name, key) + 'c(' + ','.join([str(x) for x in values[key][method_name]]) + ')')
    print()

    for key in values:
        print('norm_test_{0}_{1} <- shapiro.test(values_{0}_{1})'.format(method_name, key))
    print()

    print('if(norm_test_{0}_raw$p.value > {1} & norm_test_{0}_lemma$p.value > {1}) '.format(method_name, alpha) + '{')
    print('\teq_test_{0} <- t.test(values_{0}_raw, values_{0}_lemma, alternative = "less", paired=TRUE)'.format(method_name))
    print('\ttest_name_{} <- "t-test"'.format(method_name))
    print('} else {')
    print('\teq_test_{0} <- wilcox.test(values_{0}_raw, values_{0}_lemma, alternative = "less", paired=TRUE)'.format(method_name))
    print('\ttest_name_{} <- "wilcox"'.format(method_name))
    print('}')

    print()

    print(
        'write(\n\tpaste(' +
        ','.join([
            '"{}"'.format(method_name),
            'test_name_{}'.format(method_name),
            'sprintf("%0.{1}f", round(eq_test_{0}$p.value, digits = {1}))'.format(method_name, p_value_digit_count),
            'if(eq_test_{}$p.value < 0.05) \'Áno\' else \'Nie\''.format(method_name),
            'sep = ","'
        ])
        + '\t),'
    )
    print('\tfile="./{}",'.format(file_name))
    print('\tappend=TRUE')
    print(')')

    print()
