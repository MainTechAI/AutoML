from auto import ModelSelection
import pandas as pd
from sklearn.datasets import make_classification

used_algo = {
    'AdaBoost': True,
    'XGBoost': True,
    'MLP': False,
    'HistGB': False,
    'Ridge': False,
    'LinearSVC': True,
    'PassiveAggressive': False,
    'LogisticRegression': False,
    'LDA': True,
    'QDA': False,
    'Perceptron': True,
    'SVM': True,
    'RandomForest': True,
    'xRandTrees': True,
    'DecisionTree': True,
    'SGD': False,
    'KNeighbors': True,
    'NearestCentroid': False,
    'GaussianProcess': False,
    'GaussianNB': True,
    'Bagging(SVС)': False,

    'LabelSpreading': False,  # 2+ 3+- (crushed at some point "n_neighbors <= n_samples")
    'BernoulliNB': False,  # 2+ 3-

    'DBN': False,  # --
    'FactorizationMachine': False,  # --
    'PolynomialNetwork': False,  # --
    'ELM': False,  # --
}

args_default = {
    'experiment_name': 'exp_resampling_ds',
    'duration': 200,
    'min_accuracy': 0.3,
    'max_model_memory': 10485760,
    'max_prediction_time': 400,
    'max_train_time': 30,
    'used_algorithms': used_algo,
    'metric': 'accuracy',  # accuracy, f1_micro, f1_macro
    'validation': '10 fold CV',
    'iterations': 50,
    'resampling': None
}

test_cases = [1, 2, 3, 4, 5]
resampling_types = [None, 'under', 'over', 'combined']
args = args_default.copy()

if __name__ == "__main__":
    for test_case in test_cases:
        if test_case == 1:
            # qsar-biodeg dataset. Data set containing values for 41 attributes (molecular descriptors) used to classify
            # 1055 chemicals into 2 classes (ready and not ready biodegradable).
            # Only numerical features.
            # None       RandomForest  0.881  [('RB', 355), ('NRB', 699)]
            # 'under'    KNN           0.980  [('NRB', 412), ('RB', 355)]
            # 'over'     SVM           0.900  [('RB', 699), ('NRB', 699)]
            # 'combined' XGBoost       0.995  [('NRB', 446), ('RB', 555)]
            DS_path = r'C:\Users\Trogwald\Desktop\push_it\AutoML\experiments\DS_CD\2 classes\biodeg.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, 0:41]
            y = DS[:, 41]
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case) + '_' + str(res_type)
                args['resampling'] = res_type
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, num_features=list(range(0, 41)))
                MS.save_results(n_best='All')

        if test_case == 2:
            # Fisher's Iris data set. 3 classes.
            # The dataset is balanced
            # None       KNeighbors_0.966  ('Iris-setosa', 49), ('Iris-versicolor', 50), ('Iris-virginica', 50)
            # 'under'    SVM_1.0           ('Iris-setosa', 49), ('Iris-versicolor', 47), ('Iris-virginica', 47)
            # 'over'     KNeighbors_0.966  ('Iris-setosa', 50), ('Iris-versicolor', 50), ('Iris-virginica', 50)
            # 'combined' SVM_1.0           ('Iris-setosa', 50), ('Iris-versicolor', 45), ('Iris-virginica', 44)
            DS_path = r'C:\Users\Trogwald\Desktop\push_it\AutoML\experiments\DS_CD\3 classes\iris.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, 0:4]
            y = DS[:, 4]
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case) + '_' + str(res_type)
                args['resampling'] = res_type
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, num_features=list(range(0, 4)))
                MS.save_results(n_best='All')

        if test_case == 3:
            # Synthetic dataset. 5 classes, 400 rows, only numeric features.
            # None       AdaBoost_0.787                [(2, 179), (4, 118), (0, 41), (1, 42), (3, 20)]
            # 'under'    KNeighbors_0.9436603773584906 [(0, 16),  (1, 20), (2, 138), (3, 20), (4, 69)]
            # 'over'     KNeighbors_0.8718547486033519 [(2, 179), (4, 179), (0, 179), (1, 179), (3, 179)
            # 'combined' KNeighbors_0.960              [(0, 149), (1, 156), (2, 69), (3, 165), (4, 83)]
            x, y = make_classification(n_samples=400, n_features=5, n_informative=4, n_redundant=0, n_repeated=0,
                                       n_classes=5, n_clusters_per_class=2, weights=[0.1, 0.1, 0.45, 0.05, 0.3],
                                       class_sep=0.8, random_state=42)
            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case) + '_' + str(res_type)
                args['resampling'] = res_type
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, num_features=list(range(0, 5)))
                MS.save_results(n_best='All')

        if test_case == 4:
            # German credit dataset. This dataset classifies people described by a set of attributes
            # as good or bad credit risks. 2 classes. 1000 rows. 7 numeric and 14 categorical features.
            # There are both numerical and categorical features
            # None       XGBoost_0.8                   [('bad', 300), ('good', 699)]
            # 'under'    KNeighbors_0.9292929292929293 [('bad', 300), ('good', 194)]
            # 'over'     XGBoost_0.8785714285714286    [('bad', 699), ('good', 699)]
            # 'combined' KNeighbors_0.97               [('bad', 291), ('good', 206)]
            DS_path = r'C:\Users\Trogwald\Desktop\push_it\AutoML\experiments\DS_CD\2 classes\credit-g.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, 0:20]
            y = DS[:, 20]

            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case) + '_' + str(res_type)
                args['resampling'] = res_type
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y,
                       num_features=[1, 4, 7, 10, 12, 15, 17],
                       cat_features=[0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])
                MS.save_results(n_best='All')

        if test_case == 5:
            # German credit dataset (modified). This dataset classifies people described by a set of attributes
            # as good or bad credit risks. 2 classes. 1000 rows. 7 numeric and 14 categorical features.
            # There are only categorical features
            # None        [('bad', 300), ('good', 699)]
            # 'under'     [('bad', 300), ('good', 194)]
            # 'over'      [('bad', 699), ('good', 699)]
            # 'combined'  [('bad', 291), ('good', 206)]
            DS_path = r'C:\Users\Trogwald\Desktop\push_it\AutoML\experiments\DS_CD\2 classes\credit-g.csv'
            DS = pd.read_csv(DS_path, skiprows=0).values
            x = DS[:, [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]]
            y = DS[:, 20]

            for res_type in resampling_types:
                args['experiment_name'] = args_default['experiment_name'] + str(test_case) + '_' + str(res_type)
                args['resampling'] = res_type
                MS = ModelSelection(**args)
                MS.fit(x=x, y=y, cat_features=list(range(0, x.shape[1])))
                MS.save_results(n_best='All')
