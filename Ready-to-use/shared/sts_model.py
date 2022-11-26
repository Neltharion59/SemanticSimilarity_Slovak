from joblib import load
from json import loads
from nltk import download
import numpy as np
from shared.sts_method_pool import sts_method_pool


class STSModel:
    def __init__(self, model_name):
        download('omw-1.4')
        self.model = load('./shared/models/{}.jolib'.format(model_name))
        with open('./shared/models/{}.json'.format(model_name), 'r') as file:
            inputs = loads(file.read())['inputs']

        self.feature_methods = []
        for feature in inputs:
            matching_methods = [x for x in sts_method_pool[feature['method_name']] if x.args_match(feature['args'])]

            if len(matching_methods) != 1:
                raise ValueError('{} matches for {}'.format(len(matching_methods), feature))

            self.feature_methods.append(matching_methods[0])
        pass

    def calculate_sts(self, text1, text2):
        feature_values = [feature_method.predict(text1, text2, None) for feature_method in self.feature_methods]
        predicted_value = self.model.predict(np.array(feature_values, dtype=np.int32).reshape(1, -1))
        return predicted_value[0]
