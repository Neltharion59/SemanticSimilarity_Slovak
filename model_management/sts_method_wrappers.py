class STSMethod:
    method_id_seed = 1

    def __init__(self, method_name, method, args):
        self.id = STSMethod.method_id_seed
        STSMethod.method_id_seed = STSMethod.method_id_seed + 1
        self.args = args

        self.name = self.generate_name(method_name)
        self.method = method

    def predict(self, text1, text2, sts_method_pool):
        return self.method(text1, text2, self.args)

    def predict_mass(self, text1_array, text2_array, sts_method_pool):
        for x, y in zip(text1_array, text2_array):
            yield self.predict(x, y, sts_method_pool)

    def generate_name(self, method_name):
        string = method_name + "___"
        i = 1
        for arg in self.args:
            string = string + ("" if i == 1 else "_") + arg + "-" + str(self.args[arg])
            i = i + 1
        return string


class STSModel(STSMethod):
    def __init__(self, method_name, method, args, input_names, train_method):
        STSMethod.__init__(self, method_name, method, args)
        self.input_method_names = input_names
        self.train_method = train_method
        self.trained = False

    def predict(self, text1, text2, sts_method_pool):
        if not self.trained:
            print("STSModel::predict(str, str) of {} could not predict because it is not trained yet".format(self.name))
            return None

        results = []
        for method_name in self.input_method_names:
            if method_name not in sts_method_pool:
                print("STSModel::predict(str, str) of {} could not find {} in method pool".format(self.name, method_name))
                return None
            else:
                results.append(sts_method_pool[method_name].predict(text1, text2, sts_method_pool))
        return self.predict_model_input(results)

    def predict_model_input(self, input_values):
        if not self.trained:
            print("STSModel::predict_model_input(list) of {} could not predict because it is not trained yet".format(self.name))
            return None

        if len(input_values) != len(self.input_method_names):
            print("STSModel::predict_model_input(list) of {} could not predict because received {}/{} required input values".format(self.name, len(input_values), len(self.input_method_names)))
            return None

        return self.method.predict([input_values])[0]

    def train(self, x_train, x_test, y_train, y_test):
        self.train_method(x_train, x_test, y_train, y_test, self.method)
        self.trained = True
