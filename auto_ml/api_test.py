from auto import ModelSelection
import pandas as pd

used_algo = {
    'AdaBoost': True,
    'XGBoost': True,
    'Bagging(SVC)': True,
    'MLP': True,
    'HistGB': True,
    'Ridge': False,
    'LinearSVC': True,
    'PassiveAggressive': False,
    'LogisticRegression': False,
    'LDA': False,
    'QDA': False,
    'Perceptron': False,
    'SVM': True,
    'RandomForest': True,
    'xRandTrees': True,
    'ELM': False,
    'DecisionTree': False,
    'SGD': False,
    'KNeighbors': False,
    'NearestCentroid': False,
    'GaussianProcess': False,
    'LabelSpreading': False,
    'BernoulliNB': False,
    'GaussianNB': False,
    'DBN': False,
    'FactorizationMachine': False,
    'PolynomialNetwork': False,
}

MS = ModelSelection(
    experiment_name='experiment_api_test',
    duration=40,
    min_accuracy=0.5,
    max_model_memory=10485760,
    max_prediction_time=400,
    max_train_time=30,
    used_algorithms=used_algo,
    metric='accuracy',
    validation='10 fold CV',
    iterations=40,
)

DS_path = r'C:\Users\path\to\dataset.csv'
DS = pd.read_csv(DS_path, skiprows=0).values

if __name__ == "__main__":
    MS.fit(
        x=DS,
        y=DS[:, 41],  # TODO Expected: [0 1], got ['NRB' 'RB'] XGBoost
        num_features=list(range(0, 41)),
        # cat_features=cat_cols,
        # txt_features=txt_cols,
    )
    MS.save_results(n_best='All')
