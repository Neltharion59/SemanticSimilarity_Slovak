class STSMethod:
    method_id_seed = 1

    def __init__(self, method_name, method, args):
        self.id = STSMethod.method_id_seed
        STSMethod.method_id_seed = STSMethod.method_id_seed + 1
        self.args = args

        self.name = self.generate_name(method_name)
        self.method = method

    def predict(self, text1, text2):
        return self.method(text1, text2, self.args)

    def predict_mass(self, text1_array, text2_array):
        for x, y in zip(text1_array, text2_array):
            yield self.predict(x, y)

    def generate_name(self, method_name):
        string = method_name + "__"
        for arg in self.args:
            string = string + "_" + arg + "-" + self.args[arg]
        return string
