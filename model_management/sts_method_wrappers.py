# Library-like script providing classes to wrap STS methods and models.

# Class to wrap a simple STS method. Allows for convenient and unified processing of various methods
class STSMethod:
    # Static way to uniquely identify each instance of this method
    method_id_seed = 1

    # Constructor
    # Params: str, func<... -> float>, dict
    # Return: STSMethod
    def __init__(self, method_name, method, args):
        # Generate id for this instance and increment the value of id generator
        self.id = STSMethod.method_id_seed
        STSMethod.method_id_seed = STSMethod.method_id_seed + 1
        # Assign attribute - Parameters of STS method
        self.args = args
        self.method_name = method_name
        # Generate name of this method - based on the method and param configuration
        self.name = self.generate_name(method_name)
        # Assign attribute - Function that calculates STS
        self.method = method

    # Predict STS for given text pair
    # Params: str, str, dict<str, STSMethod>
    # Return: float
    def predict(self, text1, text2, _):
        return self.method(text1, text2, self.args)

    # Predict STS for given lists of texts
    # Params: list<str>, list<str>, dict<str, STSMethod>
    # Return: float...
    def predict_mass(self, text1_array, text2_array, sts_method_pool):
        for x, y in zip(text1_array, text2_array):
            yield self.predict(x, y, sts_method_pool)

    # Generate name of this method based on method and param configuration
    # Params: str
    # Return: str
    def generate_name(self, method_name):
        string = method_name + "___"
        i = 1
        # Use each param to create the name
        for arg in self.args:
            string = string + ("" if i == 1 else "_") + arg + "-" + str(self.args[arg])
            i = i + 1

        return string


# Class to wrap an aggregation STS method - model. Allows for convenient and unified processing of various
# aggregation methods.
# Inherits from wrapper for simple STS method for unified processing.
class STSModel(STSMethod):
    # Constructor
    # Params: str, func<... -> float>, dict, list<str>, func
    # Return: STSModel
    def __init__(self, method_name, method, args, input_names, train_method):
        # Make use of super constructor
        STSMethod.__init__(self, method_name, method, args)
        # Assign attribute - Names of methods values of which are used to feed this aggregation method
        self.input_method_names = input_names
        # Assign attribute - Function used to train wrapped aggregation method
        self.train_method = train_method
        # Assign attribute - model is not trained right now, so it cannot predict yet
        self.trained = False

    # If trained and has access to all input methods, wrapped method will predict STS of given pair of texts.
    # Params: str, str, dict<str, STSMethod>
    # Return: float/None
    def predict(self, text1, text2, sts_method_pool):
        # If model is not trained yet, it cannot predict
        if not self.trained:
            print("STSModel::predict(str, str) of {} could not predict because it is not trained yet".format(self.name))
            return None

        # Predict values of all input methods. If we don't have access to any of them, quit,
        # as we cannot predict without them.
        results = []
        # Loop over all input methods and predict their values
        for method_name in self.input_method_names:
            # If we don't have access to any of the methods, we quit
            if method_name not in sts_method_pool:
                print("STSModel::predict(str, str) of {} could not find {} in method pool".format(self.name, method_name))
                return None
            # Otherwise, let's calculate the STS value for this pair
            else:
                results.append(sts_method_pool[method_name].predict(text1, text2, sts_method_pool))

        return self.predict_model_input(results)

    # Predict STS value using input vector of calculated input values.
    # Params: list<float>
    # Return: float/None
    def predict_model_input(self, input_values):
        # If model is not trained yet, it cannot predict
        if not self.trained:
            print("STSModel::predict_model_input(list) of {} could not predict because it is not trained yet".format(self.name))
            return None

        # If given input vector does not have the same length as expected, let's quit
        if len(input_values) != len(self.input_method_names):
            print("STSModel::predict_model_input(list) of {} could not predict because received {}/{} required input values".format(self.name, len(input_values), len(self.input_method_names)))
            return None

        # Predict the value and return it
        return round(self.method.predict([input_values])[0], 2)

    # Train the wrapped aggregating method. Return metrics of the trained model on testing data.
    # Params: DataFrame, DataFrame, DataFrame, DataFrame
    # Return: dict<str, float>
    def train(self, x_train, x_test, y_train, y_test):
        # Train the wrapped aggregation method
        result = self.train_method(x_train, x_test, y_train, y_test, self.method)
        # Remember that the model is trained
        self.trained = True
        # Return model's metrics
        return result
