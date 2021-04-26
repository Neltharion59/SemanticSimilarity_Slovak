from sklearn.linear_model import LinearRegression
from complex_similarity_methods.regression_methods_core import prepare_training_data, train_n_test
from model_management.sts_method_wrappers import STSModel
from model_management.sts_method_pool import sts_method_pool
from model_management.model_persistence import persist_sklearn_model, load_sklearn_model, get_model_id

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
print(linear_regression_model.name)

x_train, x_test, y_train, y_test = prepare_training_data("dataset_sick_all_sk.txt", linear_regression_model.input_method_names)
linear_regression_model.train(x_train, x_test, y_train, y_test)

persist_sklearn_model("dataset_sick_all_sk.txt", linear_regression_model)

loaded_model = load_sklearn_model(1)
print(sts_method_pool.keys())
pred = loaded_model.predict_mass(["kto dnes vidieť"], ["pomaranč dnes chodiť"], sts_method_pool)
print(list(pred))

print("model_name:" + linear_regression_model.name)
print("loade_name:" + loaded_model.name)
model_id = get_model_id("dataset_sick_all_sk.txt", loaded_model)

print(model_id)
model_id = get_model_id("dataset_sick_all_sk_lemma.txt", loaded_model)
print(model_id)
