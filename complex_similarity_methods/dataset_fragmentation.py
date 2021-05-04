from random import shuffle

import numpy as np


class FragmentedDatasetCV:
    def __init__(
            self,
            inputs,
            gold_values,
            k_fold
    ):
        table_with_data = [input['values'] for input in inputs]

        # Split to classes for stratification
        class_grouped_values = []
        class_grouped_labels = []
        class_count = 5
        for i in range(class_count):
            class_grouped_values.append([])
            class_grouped_labels.append([])
        for i in range(len(gold_values)):
            index = min(4, int(gold_values[i] * class_count))
            class_grouped_values[index].append([table_with_data[j][i] for j in range(len(table_with_data))])
            class_grouped_labels[index].append(gold_values[i])
        for i in range(class_count):
            temp = list(zip(class_grouped_values[i], class_grouped_labels[i]))
            shuffle(temp)
            class_grouped_values[i], class_grouped_labels[i] = zip(*temp)
            class_grouped_values[i] = list(class_grouped_values[i])
            class_grouped_labels[i] = list(class_grouped_labels[i])

        # Split to dataset fragments
        #self.validation_data = SingleDatasetFragment([], [])
        self.folds = []
        for k in range(k_fold):
            self.folds.append(SingleDatasetFragment([], []))
        for i in range(class_count):
            #index_range = (0, int(len(class_grouped_values[i]) * split_ratio.validation_ratio))


            #self.validation_data.labels = self.validation_data.labels + class_grouped_labels[i][index_range[0]:index_range[1]]
            #self.validation_data.features = self.validation_data.features + class_grouped_values[i][index_range[0]:index_range[1]]

            #index_range = (0, int(len(class_grouped_values[i]) * split_ratio.validation_ratio))

            train_size = len(class_grouped_values[i])
            batch_size = int(train_size / k_fold)
            index_range = (0, batch_size)
            # print('-' * 40)
            # print(train_size)
            # print(index_range)
            # print(batch_size)

            for k in range(k_fold):
                self.folds[k].labels = self.folds[k].labels + class_grouped_labels[i][index_range[0]:index_range[1]]
                self.folds[k].features = self.folds[k].features + class_grouped_values[i][index_range[0]:index_range[1]]
                index_range = (index_range[1], train_size if k == (k_fold - 1) else (index_range[1] + batch_size))

        # for k in range(k_fold):
        #     print('Fold ', k)
        #     print('\t{}x{}'.format(len(self.folds[k].features), len(self.folds[k].features[0])))
        #     print('\tGold: {}'.format(len(self.folds[k].labels)))
        # exit()

    def produce_split_dataset(self, fold_index):
        test_fragment = SingleDatasetFragment(self.folds[fold_index].labels, self.folds[fold_index].features)
        train_fragment = SingleDatasetFragment([], [])
        for k in range(len(self.folds)):
            if k == fold_index:
                continue

            train_fragment.labels = train_fragment.labels + self.folds[k].labels
            train_fragment.features = train_fragment.features + self.folds[k].features

        return SingleFoldDatasetFragment(train_fragment, test_fragment)


class SingleDatasetFragment:
    def __init__(self, labels, features):
        self.labels = labels
        self.features = features


class SingleFoldDatasetFragment:
    def __init__(self, train_fragment, test_fragment):
        self.train = train_fragment
        self.test = test_fragment

    def produce_sklearn_ready_data(self):
        skl_ready_labels = np.array(self.train.labels)
        skl_ready_features = np.array(self.train.features)
        skl_ready_train = SingleDatasetFragment(skl_ready_labels, skl_ready_features)

        skl_ready_labels = np.array(self.test.labels)
        skl_ready_features = np.array(self.test.features)
        skl_ready_test = SingleDatasetFragment(skl_ready_labels, skl_ready_features)

        return SingleFoldDatasetFragment(skl_ready_train, skl_ready_test)


class FragmentedDatasetSuper:
    def __init__(self, available_methods, gold_values, split_ratio):
        dataset_size = len(available_methods[list(available_methods.keys())[0]][0]['values'])

        for method_name in available_methods:
            for method_config in available_methods[method_name]:
                if len(method_config['values']) != dataset_size:
                    raise ValueError('Value count inconsistent')
        if len(gold_values) != dataset_size:
            raise ValueError('Gold value count inconsistent')

        split_index = int(split_ratio.train_ratio * dataset_size)

        indices = list(range(dataset_size))
        shuffle(indices)

        reordered_gold_train, reordered_gold_valid = [gold_values[i] for i in indices[:split_index]], [gold_values[i] for i in indices[split_index:]]

        reordered_train, reordered_valid = {}, {}
        for method_name in available_methods:
            reordered_train[method_name], reordered_valid[method_name] = [], []
            for method_config in available_methods[method_name]:
                reordered_train[method_name].append({
                    'args': method_config['args'],
                    'values': [method_config['values'][i] for i in indices[:split_index]]
                })
                reordered_valid[method_name].append({
                    'args': method_config['args'],
                    'values': [method_config['values'][i] for i in indices[split_index:]]
                })

        for method_name in reordered_train:
            for method_config in reordered_train[method_name]:
                if len(method_config['values']) != split_index:
                    raise ValueError('Train Value count inconsistent')
        if len(reordered_gold_train) != split_index:
            raise ValueError('Train Gold value count inconsistent')

        for method_name in reordered_valid:
            for method_config in reordered_valid[method_name]:
                if len(method_config['values']) != dataset_size - split_index:
                    raise ValueError('Validation Value count inconsistent')
        if len(reordered_gold_valid) != dataset_size - split_index:
            raise ValueError('Validation Gold value count inconsistent')

        self.Train = SingleDatasetFragment(reordered_gold_train, reordered_train)
        self.Validation = SingleDatasetFragment(reordered_gold_valid, reordered_valid)
