from sklearn.linear_model import LinearRegression
from complex_similarity_methods.regression_methods_core import prepare_training_data, train_n_test
from model_management.sts_method_wrappers import STSModel
from model_management.sts_method_pool import sts_method_pool

methods = [
    "hamming___normalization_strategy-shorter",
    "wu_palmer___sentence_merge_strategy-all_to_all_synset_strategy-all_synsets_wordnet-slk",
    "path___sentence_merge_strategy-all_to_all_synset_strategy-first_synsets_wordnet-slk",
    "leacock_chodorow___sentence_merge_strategy-all_to_all_synset_strategy-first_synsets_wordnet-slk"
]
args = {
    'n_jobs': -1
}

linear_regression_model = STSModel("linear_regression", LinearRegression(**args), args, methods, train_n_test)

x_train, x_test, y_train, y_test = prepare_training_data("dataset_sick_all_sk.txt", linear_regression_model.input_method_names)
linear_regression_model.train(x_train, x_test, y_train, y_test)

pred = linear_regression_model.predict("kto dnes vidieť", "pomaranč dnes chodiť", sts_method_pool)
print(pred)
pred = linear_regression_model.predict("kto dnes vidieť", "pomaranč dnes chodiť", sts_method_pool)
print(pred)
pred = linear_regression_model.predict_mass(["kto dnes vidieť"], ["pomaranč dnes chodiť"], sts_method_pool)
print(list(pred))
