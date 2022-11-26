# Library-like script providing pool of all method types and possible hyperparameter values.
# Used in optimization.

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process.kernels import *

# Prepare list of regression sklearn models to be added to pool
model_types = [
    {
        "name": "linear_regression",
        "model": LinearRegression,
        "args": {
            'fit_intercept': [True, False],
            'normalize': [True, False],
        }
    },
    {
        "name": "support_vector_regression",
        "model": SVR,
        "args": {
            # Most significant ones
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'] + [y/(10 ** x) for x in range(1, 4) for y in [1, 2, 5]],
            'C': [y/(10 ** x) for x in [0, 1] for y in [1, 2, 3, 5, 7, 9]] + [10.0],

            # Less significant ones
            'degree': list(map(lambda x: x * 2 + 1, range(1, 10))),
            'epsilon': [y/(10 ** x) for x in range(1, 3) for y in [1, 2, 3]] + [0.05, 0.07, 0.09],
            'shrinking': [True, False],
            'max_iter': [x for x in range(100, 200)],
            'coef0': [0] + [y/(10 ** x) for x in range(1, 4) for y in [1, 2, 3]]
        }
    },
    {
        "name": "decision_tree_regression",
        "model": DecisionTreeRegressor,
        "args": {
            'splitter': ['random', 'best'],
            'max_depth': list(range(5, 20)) + [None],
            'min_samples_split': list(range(2, 50)),
            'min_samples_leaf': list(range(1, 30)),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_leaf_nodes': list(range(10, 50))
        }
    },
    {
        "name": "bayesan_ridge_regression",
        "model": BayesianRidge,
        "args": {
            'n_iter': [50 * x for x in range(1, 7)],
            'alpha_1': [x * (10 ** -y) for x in [1, 2, 5] for y in [4, 5, 6]],
            'alpha_2': [x * (10 ** -y) for x in [1, 2, 5] for y in [4, 5, 6]],
            'lambda_1': [x * (10 ** -y) for x in [1, 2, 5] for y in [4, 5, 6]],
            'lambda_2': [x * (10 ** -y) for x in [1, 2, 5] for y in [4, 5, 6]],
            'alpha_init': [None] + [x * (10 ** -y) for x in [1, 2, 5] for y in [1, 2, 3]]
        }
    },
    # Was not used in the end, as model-persisting libraries had trouble with kernels
    {
        "name": "gaussian_process_regression",
        "model": GaussianProcessRegressor,
        "args": {
            'kernel':
                [WhiteKernel(), DotProduct(), Matern(), RationalQuadratic(), ExpSineSquared(), RBF()],
            'alpha': [x * (10 ** -y) for x in [1, 2, 5] for y in [1, 2, 3]],
            'n_restarts_optimizer': [0, 1, 2]
        }
    },
    {
        "name": "random_forest_regression",
        "model": RandomForestRegressor,
        "args": {
            'n_estimators': [x * y for x in [10, 100] for y in [1, 2, 3]] + [50],
            'max_depth': list(range(5, 20)) + [None],
            'min_samples_split': list(range(2, 50)),
            'min_samples_leaf': list(range(1, 30)),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_leaf_nodes': list(range(2, 10)),
            'n_jobs': [-1]
        }
    },
    {
        "name": "gradient_boosting_regression",
        "model": GradientBoostingRegressor,
        "args": {
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'n_estimators': [100 * y for y in [2, 3, 4]],
            'max_depth': list(range(5, 20)) + [None],
            'min_samples_split': list(range(2, 50)),
            'min_samples_leaf': list(range(1, 30)),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_leaf_nodes': list(range(2, 50))
        }
    }
]
# Typically, we only use one model at a time during optimization,
# so if we want to optimize another model type, change it here.
model_type_index = 2
print(model_types[model_type_index]['name'])
model_types = [model_types[model_type_index]]
