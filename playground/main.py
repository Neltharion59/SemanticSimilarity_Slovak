from sklearn.linear_model import LinearRegression
from complex_similarity_methods.regression_methods import prepare_training_data, train_n_test

x_train, x_test, y_train, y_test = prepare_training_data("dataset_sick_all_sk.txt", [
    "hamming___normalization_strategy-shorter",
    "wu_palmer___sentence_merge_strategy-all_to_all_synset_strategy-all_synsets_wordnet-slk",
    "path___sentence_merge_strategy-all_to_all_synset_strategy-first_synsets_wordnet-slk",
    "leacock_chodorow___sentence_merge_strategy-all_to_all_synset_strategy-first_synsets_wordnet-slk"
])
linear_regression = LinearRegression(n_jobs=-1)
train_n_test(x_train, x_test, y_train, y_test, linear_regression)
