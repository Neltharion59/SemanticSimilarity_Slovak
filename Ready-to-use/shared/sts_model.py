import os

import numpy as np
from joblib import load
from json import loads
from nltk import download
from shared.sts_method_pool import sts_method_pool


class STSModel:
    def __init__(self, model_name):
        download('omw-1.4')
        self.model = load(os.path.join(os.path.dirname(__file__), 'models/{}.jolib'.format(model_name)))
        with open(os.path.join(os.path.dirname(__file__), 'models/{}.json'.format(model_name)), 'r') as file:
            config = loads(file.read())

        self.features = []
        for feature in config['inputs']:
            matching_methods = [x for x in sts_method_pool[feature['method_name']] if x.args_match(feature['args'])]

            if len(matching_methods) != 1:
                raise ValueError('{} matches for {}'.format(len(matching_methods), feature))

            self.features.append(matching_methods[0])

        self.type = ' '.join([x[0].upper() + x[1:] for x in config['type'].split('_')])
        pass

    def predict(self, text1, text2):
        feature_values = [feature_method.predict(text1, text2, None) for feature_method in self.features]
        predicted_value = self.model.predict(np.array(feature_values, dtype=np.int32).reshape(1, -1))
        return predicted_value[0]
