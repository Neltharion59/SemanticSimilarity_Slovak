from random import shuffle


class FragmentedDatasetCV:
    def __init__(self, inputs, gold_values, split_ratio, k_fold):
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
        self.validation_data = SingleDatasetFragment([], [])
        self.folds = []
        for k in range(k_fold):
            self.folds.append(SingleDatasetFragment([], []))
        for i in range(class_count):
            index_range = (0, int(len(class_grouped_values[i]) * split_ratio.validation_ratio))

            self.validation_data.labels = self.validation_data.labels + class_grouped_labels[i][index_range[0]:index_range[1]]
            self.validation_data.features = self.validation_data.features + class_grouped_values[i][index_range[0]:index_range[1]]

            index_range = (0, int(len(class_grouped_values[i]) * split_ratio.validation_ratio))
            train_size = len(class_grouped_values[i]) - index_range[1]
            batch_size = int(train_size / k_fold)

            for k in range(k_fold):
                index_range = (index_range[1], min(index_range[1] + batch_size, len(class_grouped_values[i])))
                self.folds[k].labels = self.folds[k].labels + class_grouped_labels[i][index_range[0]:index_range[1]]
                self.folds[k].features = self.folds[k].features + class_grouped_values[i][index_range[0]:index_range[1]]


class SingleDatasetFragment:
    def __init__(self, labels, features):
        self.labels = labels
        self.features = features

