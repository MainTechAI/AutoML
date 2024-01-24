from auto import ModelSelection
import pandas as pd
from sklearn.datasets import make_classification

used_algo = {
    'AdaBoost': True,
    'XGBoost': False,
    'MLP': True,
    'HistGB': False,
    'Ridge': False,
    'LinearSVC': False,
    'PassiveAggressive': False,
    'LogisticRegression': False,
    'LDA': False,
    'QDA': False,
    'Perceptron': False,
    'SVM': True,
    'RandomForest': True,
    'xRandTrees': False,
    'DecisionTree': False,
    'SGD': False,
    'KNeighbors': True,
    'NearestCentroid': False,
    'GaussianProcess': False,
    'GaussianNB': False,
    'Bagging(SVС)': False,
    'LabelSpreading': False,
    'BernoulliNB': False,
    'DBN': False,
    'FactorizationMachine': False,
    'PolynomialNetwork': False,
    'ELM': False,
}

args_default = {
    'experiment_name': 'exp_metalearnig_ds',
    'duration': 200,
    'min_accuracy': 0.3,
    'max_model_memory': 10485760,
    'max_prediction_time': 400,
    'max_train_time': 30,
    'used_algorithms': used_algo,
    'metric': 'accuracy',
    'validation': '10 fold CV',
    'iterations': 25,
    'resampling': None,
    'metalearning': True,
    'mlr_n': 5
}

test_cases = [1, 2, 3, 4, 5]
resampling_types = [None]
args = args_default.copy()

if __name__ == "__main__":
    for test_case in test_cases:
        if test_case == 1:
            # qsar-biodeg dataset. Data set containing values for 41 attributes (molecular descriptors) used to classify
            # 1055 chemicals into 2 classes (ready and not ready biodegradable).
            # Only numerical features.
            DS_path = r'.\experiments\DS_CD\2 classes\biodeg.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, 0:41]
            y = DS[:, 41]
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case)
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, num_features=list(range(0, 41)))
                MS.save_results(n_best='All')

        if test_case == 2:
            # Fisher's Iris data set. 3 classes.
            # The dataset is balanced
            DS_path = r'.\experiments\DS_CD\3 classes\iris.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, 0:4]
            y = DS[:, 4]
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case)
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, num_features=list(range(0, 4)))
                MS.save_results(n_best='All')

        if test_case == 3:
            # Synthetic dataset. 5 classes, 400 rows, only numeric features.
            x, y = make_classification(n_samples=400, n_features=5, n_informative=4, n_redundant=0, n_repeated=0,
                                       n_classes=5, n_clusters_per_class=2, weights=[0.1, 0.1, 0.45, 0.05, 0.3],
                                       class_sep=0.8, random_state=42)
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case)
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y.astype(str), num_features=list(range(0, 4)), cat_features=[4])
                MS.save_results(n_best='All')

        if test_case == 4:
            # German credit dataset. This dataset classifies people described by a set of attributes
            # as good or bad credit risks. 2 classes. 1000 rows. 7 numeric and 14 categorical features.
            # There are both numerical and categorical features
            DS_path = r'.\experiments\DS_CD\2 classes\credit-g.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, 0:20]
            y = DS[:, 20]
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case)
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y,
                       num_features=[1, 4, 7, 10, 12, 15, 17],
                       cat_features=[0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])
                MS.save_results(n_best='All')

        if test_case == 5:
            # German credit dataset (modified). This dataset classifies people described by a set of attributes
            # as good or bad credit risks. 2 classes. 1000 rows. 7 numeric and 14 categorical features.
            # There are only categorical features
            DS_path = r'.\experiments\DS_CD\2 classes\credit-g.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]]
            y = DS[:, 20]
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case)
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, cat_features=list(range(0, x.shape[1])))
                MS.save_results(n_best='All')
