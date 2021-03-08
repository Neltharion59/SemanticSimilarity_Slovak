from model_management.sts_method_value_persistor import get_persisted_method_values


class Dataset:
    def __init__(self, name, dataset_names):
        self.name = name
        self.dataset_names = dataset_names

    def get_data(self):
        available_values = {}
        for dataset_name in self.dataset_names:
            new_dataset = get_persisted_method_values(dataset_name)
            if len(set(map(lambda x: len(x), new_dataset.values()))) != 1:
                raise ValueError("class Dataset method get_data there is a dict with different value counts")
            if available_values == {}:
                available_values = new_dataset
            else:
                if len(available_values.keys()) != len(new_dataset.keys()):
                    raise ValueError("class Dataset method get_data tried to merge dicts with inequal key count")
                for method_name in available_values:
                    if method_name not in new_dataset:
                        raise ValueError("class Dataset method get_data trying to append dict with missing method to result dict")
                    available_values[method_name] = available_values[method_name] + new_dataset[method_name]
        return available_values
