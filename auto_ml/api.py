# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:17:42 2020

@author: dosto

API test
"""

from auto import ModelSelection
from utility.data import load_DS_as_df, load_CD_as_list


# three ways to specify a path string  
#DS_path=r'C:\Users\dosto\.+DATASETS\breast-w\breast-w.csv'
#DS_path='C:/Users/dosto/.+DATASETS/breast-w/breast-w.csv'
DS_path='C:\\Users\\dosto\\.+DATASETS\\breast-w\\breast-w.csv'

CD_path='C:\\Users\\dosto\\.+DATASETS\\breast-w\\column_description.csv'

DS = load_DS_as_df(DS_path).values # to numpy
CD = load_CD_as_list(CD_path)
    

used_algo = {
    'AdaBoost':True, 'XGBoost':True, 'Bagging(SVС)':True,
    'MLP':True, 'HistGB':False, 'Ridge':False,
    'LinearSVC':False, 'PassiveAggressive':False, 'LogisticRegression':False,
    'LDA':False, 'QDA':False, 'Perceptron':False,      
    'SVM':True, 'RandomForest':True, 'xRandTrees':True,
    'ELM':False, 'DecisionTree':False, 'SGD':False,
    'KNeighbors':False, 'NearestCentroid':False, 'GaussianProcess':False,
    'LabelSpreading':False, 'BernoulliNB':False, 'GaussianNB':False,
    'DBN':False, 'FactorizationMachine':False, 'PolynomialNetwork':False
}
        
ModelSelection(
               DS = DS, # TODO to numpy
               CD = CD, # TODO change to dict? cat_col=[3,4], num_col=[1,2,5]
               experiment_name = 'experiment_api_test', 
               duration = 40, 
               min_accuracy = 65.0,
               max_model_memory = 1048576, 
               max_prediction_time = 40, 
               max_train_time = 30, 
               used_algorithms = used_algo, 
               metric = 'accuracy', 
               validation = '10 fold CV', 
               saved_models_count = 'All',
               iterations = 60,
               )