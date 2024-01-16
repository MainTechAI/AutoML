from sklearn import naive_bayes, svm, linear_model, discriminant_analysis, neighbors, gaussian_process
from sklearn import tree, ensemble, semi_supervised, neural_network
from hyperopt import hp
import xgboost
import dbn
import polylearn
import numpy as np
import uuid

np.random.seed(42)


class ModelHolder:
    def __init__(self):
        self.all_models = [
            Perceptron(), Ridge(), PassiveAggressive(), LogisticRegression(), LDA(), QDA(), LinearSVC(), SVM(), SGD(),
            KNeighbors(), NearestCentroid(), GaussianProcess(), BernoulliNB(), GaussianNB(), DecisionTree(),
            BaggingSVC(), RandomForest(), xRandTrees(), AdaBoost(), HistGB(), LabelSpreading(), MLP(), XGBoost(),
            # TODO: add DummyClassifier for comparison
        ]

    def get_approved_models(self, used_algorithms):
        approved_models = []
        used_names = []
        models_verbose = []
        for name in used_algorithms:
            if used_algorithms.get(name) == True:
                used_names.append(name)

        for model in self.all_models:
            if model.short_name in used_names:
                approved_models.append(model)
                models_verbose.append(model.short_name)

        print('The following algorithms will be used in the search: ', models_verbose)
        return approved_models

    def get_all_models(self):
        models = []
        for m in self.all_models:
            models.append((m.short_name, m.get_skl_estimator()))
        return models


def unique_n(name):
    return name + uuid.uuid4().hex[:6].upper()


class SVM:
    def __init__(self, ):
        self.name = 'Support Vector Classification'
        self.short_name = 'SVM'

        self.default_parameters = {'C': 1.0, 'kernel': 'rbf', 'degree': 3,
                                   'gamma': 'scale', 'coef0': 0.0, 'shrinking': True, 'probability': False,
                                   'tol': 1e-3, 'cache_size': 200, 'class_weight': None, 'verbose': False,
                                   'max_iter': -1, 'decision_function_shape': 'ovr',
                                   'break_ties': False, 'random_state': None}

        self.scale = True

        self.search_space = {
            'name': 'SVM',
            'scale': hp.choice('SVM_scale_1', [True, False]),
            'model': svm.SVC,
            'param': hp.pchoice('SVM_kernel_type', [
                (0.65, {
                    'kernel': hp.choice('SVM_p11', ['rbf', 'sigmoid']),
                    'gamma': hp.pchoice('SVM_p12', [(0.05, 'scale'), (0.05, 'auto'),
                                                    (0.9, hp.loguniform('SVM_p121', -10.4, 2.08))]),
                    'C': hp.loguniform('SVM_p13', -3.46, 10.4),
                    'degree': 2
                }),
                (0.35, {
                    'kernel': 'poly',  # computationally heavy
                    'gamma': 'scale',
                    'C': hp.loguniform('SVM_p21', -3.46, 3),  # -3.46, 5
                    'degree': hp.choice('SVM_p22', range(2, 5))
                })
            ])
        }

    def get_skl_estimator(self, **default_parameters):
        return svm.SVC(**default_parameters)


class LinearSVC:
    def __init__(self, ):
        self.name = 'Linear Support Vector Classification'
        self.short_name = 'LinearSVC'

        self.search_space = {
            'name': 'LinearSVC',
            'scale': hp.choice('LinearSVC_scale_1', [True, False]),
            'model': svm.LinearSVC,
            'param': {
                'C': hp.loguniform('LinearSVC_p1', -3.46, 10.4),
                'tol': hp.loguniform('LinearSVC_p2', -11.5129, -2.30259),
                'dual': hp.choice('LinearSVC_p3', [True, False]),
                'max_iter': hp.choice('LinearSVC_p4', [1000, 2000, 4000, 8000])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return svm.LinearSVC(**default_parameters)


class XGBoost:
    def __init__(self, ):
        self.name = 'eXtreme Gradient Boosting'
        self.short_name = 'XGBoost'

        self.scale = None

        self.default_parameters = {
            "max_depth": 3, "learning_rate": 0.1, "n_estimators": 100,
            "verbosity": 1, "silent": None, "objective": "binary:logistic",
            "booster": 'gbtree', "n_jobs": 1, "nthread": None, "gamma": 0,
            "min_child_weight": 1, "max_delta_step": 0, "subsample": 1,
            "colsample_bytree": 1, "colsample_bylevel": 1, "colsample_bynode": 1,
            "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": 1, "base_score": 0.5,
            "random_state": 0, "seed": None, "missing": None
        }
        self.parameters_mandatory_first_check = [
            {"learning_rate": 0.018, "n_estimators": 4168},
            {"learning_rate": 0.018, "n_estimators": 4168, "subsample": 0.84,
             "max_depth": 13, "min_child_weight": 2, "colsample_bytree": 0.75,
             "colsample_bylevel": 0.58, "reg_lambda": 0.98, "reg_alpha": 1.11},
            self.default_parameters
        ]
        self.parameters_range = {
            "n_estimators": [1, 5000],
            "learning_rate": [2 ** -10, 2 ** 0],
            "subsample": [0.1, 1],
            "booster": ["gbtree", "gblinear", "dart"],
            "max_depth": [1, 15],
            "min_child_weight": [2 ** 0, 2 ** 7],
            "colsample_bytree": [0, 1],
            "colsample_bylevel": [0, 1],
            "reg_lambda": [2 ** -10, 2 ** 10],
            "reg_alpha": [2 ** -10, 2 ** 10],
        }

        # TODO: split into different models by booster type?

        self.search_space = {
            'name': 'XGBoost',
            'scale': None,
            'model': xgboost.XGBClassifier,
            'param': {
                # "n_estimators":hp.randint('XGBoost_p1', 500), # was 5000
                # "subsample":hp.uniform('XGBoost_p3', 0.1, 1),
                # "booster": hp.choice('XGBoost_p4', ["gbtree","gblinear","dart"]),
                # 'max_depth': hp.choice('XGBoost_p5',range(1,15)),
                # "min_child_weight":hp.loguniform('XGBoost_p6', 0, 4.852),
                # "colsample_bytree":hp.uniform('XGBoost_p7', 0,1),
                # "colsample_bylevel":hp.uniform('XGBoost_p8', 0,1),
                # "reg_lambda":hp.loguniform('XGBoost_p9', -6.931, 6.931),
                # "reg_alpha":hp.loguniform('XGBoost_p10', -6.931, 6.931),
                'eval_metric': 'mlogloss',
                "learning_rate": hp.loguniform('XGBoost_p2', -6.931, 0),
                'max_depth': ('XGBoost_p5', range(1, 15)),
                'gamma': ('gamma', 1, 9),
                'reg_alpha': ('reg_alpha', 40, 180, 1),
                'reg_lambda': ('reg_lambda', 0, 1),
                'colsample_bytree': ('colsample_bytree', 0.5, 1),
                'min_child_weight': ('min_child_weight', 0, 10, 1),
                'n_estimators': ('XGBoost_p1', 500),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return xgboost.XGBClassifier(**default_parameters)


class Perceptron:
    def __init__(self, ):
        self.name = 'Perceptron'
        self.short_name = 'Perceptron'

        self.scale = None

        self.default_parameters = {
            "penalty": None, "alpha": 0.0001, "fit_intercept": True,
            "max_iter": 1000, "tol": 1e-3, "shuffle": True, "verbose": 0,
            "eta0": 1.0, "n_jobs": None, "random_state": 0,
            "early_stopping": False, "validation_fraction": 0.1,
            "n_iter_no_change": 5, "class_weight": None, "warm_start": False
        }

        self.search_space = {
            'name': 'Perceptron',
            'model': linear_model.Perceptron,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.Perceptron(**default_parameters)


class Ridge:
    def __init__(self, ):
        self.name = 'Ridge regression сlassifier'
        self.short_name = 'Ridge'

        self.scale = None

        self.default_parameters = {
            "alpha": 1.0, "fit_intercept": True, "normalize": False,
            "copy_X": True, "max_iter": None, "tol": 1e-3, "class_weight": None,
            "solver": "auto", "random_state": None
        }

        self.search_space = {
            'name': 'Ridge',
            'model': linear_model.RidgeClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.RidgeClassifier(**default_parameters)


class PassiveAggressive:
    def __init__(self, ):
        self.name = 'Passive Aggressive Classifier'
        self.short_name = 'PassiveAggressive'

        self.scale = None

        self.default_parameters = {
            "C": 1.0, "fit_intercept": True, "max_iter": 1000, "tol": 1e-3,
            "early_stopping": False, "validation_fraction": 0.1,
            "n_iter_no_change": 5, "shuffle": True, "verbose": 0, "loss": "hinge",
            "n_jobs": None, "random_state": None, "warm_start": False,
            "class_weight": None, "average": False
        }

        self.search_space = {
            'name': 'PassiveAggressive',
            'model': linear_model.PassiveAggressiveClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.PassiveAggressiveClassifier(**default_parameters)


class LogisticRegression:
    def __init__(self, ):
        """
        Robust to unscaled datasets
        https://scikit-learn.org/stable/modules/linear_model.html
        """

        self.name = 'Logistic Regression'
        self.short_name = 'LogisticRegression'

        self.default_parameters = {'penalty': 'l2', 'dual': False, 'tol': 1e-4,
                                   'C': 1.0, 'fit_intercept': True,
                                   'intercept_scaling': 1, 'class_weight': None,
                                   'random_state': None, 'solver': 'lbfgs',
                                   'max_iter': 100, 'multi_class': 'auto',
                                   'verbose': 0, 'warm_start': False,
                                   'n_jobs': None, 'l1_ratio': None}

        self.scale = True

        self.search_space = {
            'name': 'LogisticRegression',
            'model': linear_model.LogisticRegression,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.LogisticRegression(**default_parameters)


class LDA:
    def __init__(self, ):
        self.name = 'Linear Discriminant Analysis'
        self.short_name = 'LDA'

        self.scale = None

        self.default_parameters = {
            "solver": 'svd',
            "shrinkage": None,
            "priors": None,
            "n_components": None,
            "store_covariance": False,
            "tol": 1e-4
        }

        self.search_space = {
            'name': 'LDA',
            'model': discriminant_analysis.LinearDiscriminantAnalysis,
            'param': hp.choice('LDA_solver', [
                {
                    'solver': hp.choice('LDA_p11', ['lsqr', 'eigen']),
                    'shrinkage': hp.choice('LDA_p12', [None, 'auto', hp.uniform('LDA_p121', 0, 1)]),
                    'tol': None,
                },
                {
                    'solver': 'svd',
                    'shrinkage': None,
                    'tol': hp.loguniform('LDA_p26', -10, 0),
                }
            ])
        }

    def get_skl_estimator(self, **default_parameters):
        return discriminant_analysis.LinearDiscriminantAnalysis(**default_parameters)


class QDA:
    def __init__(self, ):
        self.name = 'Quadratic Discriminant Analysis'
        self.short_name = 'QDA'

        self.default_parameters = {
            "priors": None,
            "reg_param": 0.,
            "store_covariance": False,
            "tol": 1.0e-4,
        }

        self.scale = None  # ???

        self.search_space = {
            'name': 'QDA',
            'model': discriminant_analysis.QuadraticDiscriminantAnalysis,
            'param': {
                'reg_param': hp.uniform('QDA1', 0.0, 1.0),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return discriminant_analysis.QuadraticDiscriminantAnalysis(**default_parameters)


class SGD:
    def __init__(self, ):
        self.name = 'SVM with SGD'
        self.short_name = 'SGD'

        self.scale = None

        self.default_parameters = {
            "loss": "hinge", "penalty": 'l2', "alpha": 0.0001, "l1_ratio": 0.15,
            "fit_intercept": True, "max_iter": 1000, "tol": 1e-3, "shuffle": True,
            "verbose": 0, "epsilon": 0.1, "n_jobs": None,
            "random_state": None, "learning_rate": "optimal", "eta0": 0.0,
            "power_t": 0.5, "early_stopping": False, "validation_fraction": 0.1,
            "n_iter_no_change": 5, "class_weight": None, "warm_start": False,
            "average": False
        }

        self.search_space = {
            'name': 'SGD',
            'model': linear_model.SGDClassifier,
            'param': None
            # sgd_loss = hp.pchoice(’loss’, [(0.50, ’hinge’), (0.25, ’log’), (0.25, ’huber’)])
            # sgd_penalty = hp.choice(’penalty’, [’l2’, ’elasticnet’])
            # sgd_alpha = hp.loguniform(’alpha’, low=np.log(1e-5), high=np.log(1) )
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.SGDClassifier(**default_parameters)


class KNeighbors:
    def __init__(self, n_rows=1000):
        # TODO: try NeighborhoodComponentsAnalysis + KNeighborsClassifier
        self.name = 'K-nearest neighbors classifier'
        self.short_name = 'KNeighbors'

        self.scale = None

        self.default_parameters = {
            "n_neighbors": 5, "weights": 'uniform', "algorithm": 'auto',
            "leaf_size": 30, "p": 2, "metric": 'minkowski',
            "metric_params": None, "n_jobs": None
        }

        self.parameters_mandatory_first_check = [
            {'n_neighbors': int(n_rows ** 0.5)},
            {'n_neighbors': 30},
            self.default_parameters
        ]

        self.hpo_results = []

        self.search_space = {
            'name': 'KNeighbors',
            'model': neighbors.KNeighborsClassifier,
            'param': {
                #            "n_neighbors":1+hp.randint('KNeighbors_p1', 50),
                "n_neighbors": hp.qloguniform('KNeighbors_p1', np.log(1), np.log(50), 1),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return neighbors.KNeighborsClassifier(**default_parameters)


class NearestCentroid:
    def __init__(self, ):
        self.name = 'Nearest centroid classifier.'
        self.short_name = 'NearestCentroid'

        self.scale = None

        self.default_parameters = {
            "metric": 'euclidean', "shrink_threshold": None
        }

        self.parameters_mandatory_first_check = [
            self.default_parameters
        ]

        self.search_space = {
            'name': 'NearestCentroid',
            'model': neighbors.NearestCentroid,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return neighbors.NearestCentroid(**default_parameters)


class GaussianProcess:
    def __init__(self, ):
        """
        1.0 * RBF(1.0)

        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
        kernels ^
        """

        self.name = 'Gaussian Process Classifier'
        self.short_name = 'GaussianProcess'

        self.scale = None

        self.default_parameters = {
            "kernel": None, "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 0, "max_iter_predict": 100,
            "warm_start": False, "copy_X_train": True, "random_state": None,
            "multi_class": "one_vs_rest", "n_jobs": None
        }

        self.search_space = {
            'name': 'GaussianProcess',
            'model': gaussian_process.GaussianProcessClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return gaussian_process.GaussianProcessClassifier(**default_parameters)


class BernoulliNB:
    def __init__(self, ):
        self.name = 'Naive Bayes classifier for multivariate Bernoulli models'
        self.short_name = 'BernoulliNB'

        self.scale = None

        self.default_parameters = {
            "alpha": 1.0,
            "binarize": .0,
            "fit_prior": True,
            "class_prior": None
        }

        self.search_space = {
            'name': 'BernoulliNB',
            'model': naive_bayes.BernoulliNB,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return naive_bayes.BernoulliNB(**default_parameters)


class GaussianNB:
    def __init__(self, ):
        self.name = 'Gaussian Naive Bayes'
        self.short_name = 'GaussianNB'

        self.scale = None

        self.default_parameters = {
            "priors": None,
            "var_smoothing": 1e-9
        }

        self.search_space = {
            'name': 'GaussianNB',
            'model': naive_bayes.GaussianNB,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return naive_bayes.GaussianNB(**default_parameters)


class DecisionTree:
    def __init__(self, ):
        self.name = 'Decision tree classifier'
        self.short_name = 'DecisionTree'

        self.scale = None

        self.default_parameters = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "max_features": None,
            "random_state": None,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
            "min_impurity_split": None,
            "class_weight": None,
            "presort": 'deprecated',
            "ccp_alpha": 0.0
        }

        self.search_space = {
            'name': 'DecisionTree',
            'model': tree.DecisionTreeClassifier,
            'param': {
                'criterion': hp.choice(unique_n('DecisionTree_criterion'), ['gini', 'entropy']),
                'splitter': hp.choice(unique_n('DecisionTree_splitter'), ["best", "random"]),
                'max_depth': hp.pchoice(unique_n('DecisionTree_max_depth'),
                                        [(0.7, None), (0.1, 2), (0.1, 3), (0.1, 4)]),
                'min_samples_split': hp.pchoice(unique_n('DecisionTree_min_samples_split'), [(0.95, 2), (0.05, 3)]),
                'min_weight_fraction_leaf': hp.pchoice(unique_n('DecisionTree_min_weight_fraction_leaf'),
                                                       [(0.95, 0.0), (0.05, 0.01)]),
                'max_features': hp.pchoice(unique_n('DecisionTree_max_features'), [(0.2, "sqrt"), (0.1, "log2"),
                                                                                   (0.1, None),
                                                                                   (0.6,
                                                                                    hp.uniform(unique_n("DT_max_f"), 0.,
                                                                                               1.))])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return tree.DecisionTreeClassifier(**default_parameters)


class BaggingSVC:
    def __init__(self, ):
        """
        bagging methods work best with strong and complex models
        (e.g., fully developed decision trees), in contrast with
        boosting methods which usually work best with weak models
        (e.g., shallow decision trees).
        """

        self.name = 'Bagging classifier'
        self.short_name = 'Bagging(SVС)'

        self.default_parameters = {
            "base_estimator": None,
            "n_estimators": 10,
            "max_samples": 1.0,
            "max_features": 1.0,
            "bootstrap": True,
            "bootstrap_features": False,

            "oob_score": False,
            "warm_start": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0
        }

        self.search_space = {
            'name': 'Bagging(SVС)',
            'model': ensemble.BaggingClassifier,
            "scale": hp.choice('Bagging(SVС)_scale', [True, False]),
            'param': {
                "bootstrap": hp.choice('Bagging(SVС)_bootstrap', [True, False]),
                "n_estimators": hp.choice('Bagging(SVС)_n_estimators', [4, 8, 16, 32, 64]),
                "max_features": hp.pchoice('Bagging(SVС)_max_features', [(0.05, 0.8), (0.15, 0.9), (0.8, 1.0)]),
                "max_samples": hp.pchoice('Bagging(SVС)_max_samples', [(0.05, 0.8), (0.15, 0.9), (0.8, 1.0)]),
                "bootstrap_features": hp.choice('Bagging(SVС)_bootstrap_features', [True, False]),
                "base_estimator": hp.choice('Bagging(SVС)_base_estimator', [
                    {
                        "model": svm.SVC,
                        "param": {
                            "kernel": 'rbf',
                            'gamma': hp.pchoice('Bagging(SVС)_p1_gamma', [(0.05, 'scale'), (0.05, 'auto'),
                                                                          (0.9,
                                                                           hp.loguniform('Bagging(SVС)_p1_gamma_sub',
                                                                                         -10.4, 2.08))]),
                            'C': hp.loguniform('Bagging(SVС)_p1_C', -3.46, 4),
                        }
                    },
                    {
                        "model": DecisionTree().search_space['model'],
                        "param": DecisionTree().search_space['param']
                    },
                    {
                        "model": LinearSVC().search_space['model'],
                        "param": LinearSVC().search_space['param']
                    }

                    # TODO you can add any other classifier

                ])

            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.BaggingClassifier(**default_parameters)


class RandomForest:
    def __init__(self, ):
        self.name = 'Random forest classifier'
        self.short_name = 'RandomForest'

        self.scale = None

        self.default_parameters = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
            "min_impurity_split": None,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None
        }

        self.parameters_range = {
            'max_features': [0.1, 0.9],
            'min_samples_leaf': [1, 20],
            'bootstrap': [True, False]
        }

        self.search_space = {
            'name': 'RandomForest',
            'model': ensemble.RandomForestClassifier,
            'param': {
                'max_features': hp.uniform('RandomForest_p1', 0.1, 0.9),
                'min_samples_leaf': 1 + hp.randint('RandomForest_p2', 20),
                'bootstrap': hp.choice('RandomForest_p3', [True, False])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.RandomForestClassifier(**default_parameters)


class xRandTrees:
    def __init__(self, ):
        self.name = 'Extra-trees classifier'
        self.short_name = 'xRandTrees'

        self.scale = None

        self.default_parameters = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
            "min_impurity_split": None,
            "bootstrap": False,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None
        }

        self.search_space = {
            'name': 'xRandTrees',
            'model': ensemble.ExtraTreesClassifier,
            'param': {
                'max_features': hp.uniform('xRandTrees_p1', 0.1, 0.9),
                'min_samples_leaf': 1 + hp.randint('xRandTrees_p2', 20),
                'bootstrap': hp.choice('xRandTrees_p3', [True, False]),
                "min_samples_split": hp.pchoice('xRandTrees_p4', [(0.95, 2), (0.05, 3), ])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.ExtraTreesClassifier(**default_parameters)


class AdaBoost:
    def __init__(self, ):
        self.name = 'Adaptive Boosting classifier'
        self.short_name = 'AdaBoost'

        self.scale = None

        self.default_parameters = {
            "base_estimator": None,
            "n_estimators": 50,
            "learning_rate": 1.,
            "algorithm": 'SAMME.R',
            "random_state": None
        }

        self.parameters_mandatory_first_check = [
            {"base_estimator": DecisionTree().get_skl_estimator(max_depth=10)}
        ]

        self.parameters_range = {
            "learning_rate": [0.01, 2.0],  # (log-scale)
            "base_estimator": DecisionTree().get_skl_estimator(
                max_depth=[1, 10])  # (10 optimal maybe need more)
        }

        self.hpo_results = []

        self.search_space = {
            'name': 'AdaBoost',
            'model': ensemble.AdaBoostClassifier,
            'param': {
                "learning_rate": hp.uniform('AdaBoost_p1', 0.01, 2.0),
                "base_estimator": {
                    "model": DecisionTree().search_space['model'],
                    "param": DecisionTree().search_space['param']
                }
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.AdaBoostClassifier(**default_parameters)


class HistGB:
    def __init__(self, ):
        """
        This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10_000).
        This estimator has native support for missing values (NaNs)
        Although it's better to use lightgbm
        """

        self.name = 'Histogram-based Gradient Boosting Classification Tree'
        self.short_name = 'HistGB'

        self.scale = None

        self.default_parameters = {
            "loss": 'auto',
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.,
            "max_bins": 255,
            "warm_start": False,
            "scoring": None,
            "validation_fraction": 0.1,
            "n_iter_no_change": None,
            "tol": 1e-7,
            "verbose": 0,
            "random_state": None
        }

        self.search_space = {
            'name': 'HistGB',
            'model': ensemble.HistGradientBoostingClassifier,
            'param': {
                "learning_rate": hp.loguniform('HistGBp1', -7, 0),
                "max_iter": 30 + hp.randint('HistGBp2', 250),
                'l2_regularization': hp.choice('HistGBp5', [0, 0.1, 0.01, 0.001]),
                'min_samples_leaf': 5 + hp.randint('HistGBp4', 30),
                'max_depth': hp.choice('HistGBp3', [None, 2 + hp.randint('HistGBp3_1', 15)]),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.HistGradientBoostingClassifier(**default_parameters)


class LabelSpreading:
    def __init__(self, ):
        self.name = 'LabelSpreading semi-supervised'
        self.short_name = 'LabelSpreading'

        self.scale = None

        self.default_parameters = {
            "kernel": 'rbf', "gamma": 20, "n_neighbors": 7, "alpha": 0.2,
            "max_iter": 30, "tol": 1e-3, "n_jobs": None
        }

        self.parameters_mandatory_first_check = [
            self.default_parameters
        ]

        self.search_space = {
            'name': 'LabelSpreading',
            'model': semi_supervised.LabelSpreading,
            'param': {
                'kernel': hp.choice('LabelSpreading_p1', ['knn', 'rbf']),
                'gamma': hp.loguniform('LabelSpreading_p2', -14, 4),  # Parameter for rbf kernel. 70.4
                'n_neighbors': 1 + hp.randint('LabelSpreading_p3', 150),  # Parameter for knn kernel
                'alpha': hp.uniform('LabelSpreading_p4', 0, 1),
                'max_iter': 2 + hp.randint('LabelSpreading_p5', 150),
                'tol': hp.loguniform('LabelSpreading_p6', -10, 0),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return semi_supervised.LabelSpreading(**default_parameters)


class MLP:
    def __init__(self, ):
        self.name = 'Multi-layer Perceptron classifier'
        self.short_name = 'MLP'
        self.scale = None
        self.default_parameters = {
            "hidden_layer_sizes": (100,), "activation": "relu",
            "solver": 'adam', "alpha": 0.0001,
            "batch_size": 'auto', "learning_rate": "constant",
            "learning_rate_init": 0.001, "power_t": 0.5, "max_iter": 200,
            "shuffle": True, "random_state": None, "tol": 1e-4,
            "verbose": False, "warm_start": False, "momentum": 0.9,
            "nesterovs_momentum": True, "early_stopping": False,
            "validation_fraction": 0.1, "beta_1": 0.9, "beta_2": 0.999,
            "epsilon": 1e-8, "n_iter_no_change": 10, "max_fun": 15000
        }

        self.parameters_mandatory_first_check = [
            self.default_parameters
        ]

        self.search_space = {
            'name': 'MLP',
            'model': neural_network.MLPClassifier,
            'scale': hp.choice('MLP_scale_1', [True, False]),
            'param': {
                'hidden_layer_sizes': hp.choice('MLP_p1', [
                    (1 + hp.randint('size11', 300)),
                    (1 + hp.randint('size21', 200), 1 + hp.randint('size22', 200)),
                    # (1+hp.randint('size31', 200),1+hp.randint('size32', 200),1+hp.randint('size33', 200)),
                ]),
                'activation': hp.choice('MLP_p2', ['identity', 'logistic', 'tanh', 'relu']),
                'solver': hp.choice('MLP_p3', ['lbfgs', 'sgd', 'adam']),
                'learning_rate': hp.choice('MLP_p4', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': hp.loguniform('MLP_p5', -9, 0),
                'max_iter': 50 + hp.randint('MLP_p6', 750),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return neural_network.MLPClassifier(**default_parameters)


# TODO: fix
class DBN:
    def __init__(self, ):
        self.name = 'Deep Belief Network Classifier'
        self.short_name = 'DBN'

        self.default_parameters = {
            "hidden_layers_structure": [100, 100],
            "activation_function": 'sigmoid',
            "optimization_algorithm": 'sgd',
            "learning_rate": 1e-3,
            "learning_rate_rbm": 1e-3,
            "n_iter_backprop": 100,
            "l2_regularization": 1.0,
            "n_epochs_rbm": 10,
            "contrastive_divergence_iter": 1,
            "batch_size": 32,
            "dropout_p": 0,
            "verbose": False
        }

        self.search_space = {
            'name': 'DBN',
            'model': dbn.SupervisedDBNClassification,
            'param': None,
        }

    def get_skl_estimator(self, **default_parameters):
       return dbn.SupervisedDBNClassification(**default_parameters)


class FactorizationMachine:
    def __init__(self, ):
        self.name = 'Factorization Machine Classifier'
        self.short_name = 'FactorizationMachine'

        self.default_parameters = {
            "degree": 2,
            "loss": 'squared_hinge',
            "n_components": 2,
            "alpha": 1,
            "beta": 1,
            "tol": 1e-6,
            "fit_lower": 'explicit',
            "fit_linear": True,
            "warm_start": False,
            "init_lambdas": 'ones',
            "max_iter": 10000,
            "verbose": False,
            "random_state": None
        }

        self.parameters_mandatory_first_check = [
            {"n_components": 1},
            {"n_components": 2},
            {"n_components": 3}
        ]

        self.search_space = {
            'name': 'FactorizationMachine',
            'model': polylearn.FactorizationMachineClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return polylearn.FactorizationMachineClassifier(**default_parameters)


class PolynomialNetwork:
    def __init__(self, ):
        self.name = 'Polynomial Network Classifier'
        self.short_name = 'PolynomialNetwork'

        self.default_parameters = {
            "degree": 2, "loss": 'squared_hinge', "n_components": 2, "beta": 1,
            "tol": 1e-6, "fit_lower": 'augment', "warm_start": False,
            "max_iter": 10000, "verbose": False, "random_state": None
        }

        self.parameters_mandatory_first_check = [
            {"degree": 2},
            {"degree": 3}
        ]

        self.search_space = {
            'name': 'PolynomialNetwork',
            'model': polylearn.PolynomialNetworkClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return polylearn.PolynomialNetworkClassifier(**default_parameters)


"""
from sklearn import dummy

class DummyClassifier:
    def __init__(self, ):

        self.name = 'Dummy Classifier'
        self.short_name = 'Dummy'

        self.scale = None

        self.default_parameters={
            "strategy":"warn",
            "random_state":None,
            "constant":None
                }

        self.parameters_mandatory_first_check=[
                {'strategy':'stratified'},
                {'strategy':'most_frequent'},
                {'strategy':'prior'},
                {'strategy':'uniform'}
                ]

    def get_skl_estimator(self, **default_parameters):
        return dummy.DummyClassifier(**default_parameters)

"""
