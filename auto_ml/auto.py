import re
from time import perf_counter
from sklearn import neural_network, ensemble, svm
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import f1_score, make_scorer
from AutoML.meta_learning import EvalParserSVM, EvalParserAdaBoost, EvalParserMLP, EvalParserRandomForest
from AutoML.meta_learning.util.utils import get_nearest_dids, extract_meta_features
from models import ModelHolder
import numpy as np
import pandas as pd
import os
import joblib
import hyperopt
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utility.util import split_val_score, cross_val_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)


class ModelSelection:
    def __init__(self, experiment_name, duration, min_accuracy,
                 max_model_memory, max_prediction_time, max_train_time,
                 used_algorithms, metric, validation, iterations,
                 resampling=None, metalearning=False, mlr_n=1, ensembling=False, n_estimators='all', max_jobs=1):
        print('Loading started')
        self.resampling = resampling
        self.metalearning = metalearning
        self.mlr_n = mlr_n
        self.ensembling = ensembling
        self.n_estimators = n_estimators
        self.status = ''

        self.original_x = None
        self.original_y = None
        self.original_num_cols = None
        self.original_cat_cols = None

        self.row_count = None
        self.columns_count = None
        self.target_column = None
        self.cat_columns = []
        self.path_to_save = None

        self.max_jobs = max_jobs
        self.CV_jobs = self.max_jobs

        self.experiment_name = experiment_name
        self.duration = duration
        self.min_accuracy = min_accuracy
        self.max_model_memory = max_model_memory
        self.max_prediction_time = max_prediction_time
        self.max_train_time = max_train_time
        self.iterations = iterations

        self.used_algorithms = used_algorithms
        self.validation = validation

        # tested: accuracy, roc_auc, balanced_accuracy
        # currently there are some problems with f1, recall, precision
        self.metric = metric
        self.metric_original = metric
        if self.metric == 'f1_micro':
            self.metric = make_scorer(f1_score, greater_is_better=True, average="micro")
        if self.metric == 'f1_macro':
            self.metric = make_scorer(f1_score, greater_is_better=True, average="macro")

        self.valtype = ''
        self.cv_splits = None

        # TODO change
        if self.validation in ["3 fold CV", "5 fold CV", "10 fold CV"]:
            if self.validation == "3 fold CV":
                self.cv_splits = 3
            elif self.validation == "5 fold CV":
                self.cv_splits = 5
            elif self.validation == "10 fold CV":
                self.cv_splits = 10
            self.valtype = 'CV'
            from sklearn import model_selection
            self.kfold = model_selection.KFold(n_splits=self.cv_splits)
        elif self.validation == "holdout":
            self.valtype = 'H'

        self.models = ModelHolder().get_approved_models(self.used_algorithms)
        print('Loading ended')
        print()

    def fit(self, x, y, num_features=[], cat_features=[], txt_features=[]):
        self.original_x = x.copy()
        self.original_y = y.copy()
        self.original_num_cols = num_features.copy()
        self.original_cat_cols = cat_features.copy()
        print('Original x shape', self.original_x.shape, 'y shape', self.original_y.shape)
        self.x = x.copy()
        self.y = y.copy()

        # If a numeric feature treated as a categorical, category_encoders will throw an error
        # AttributeError: 'numpy.ndarray' object has no attribute 'columns'
        if len(cat_features) != 0:
            from category_encoders import OrdinalEncoder
            enc = OrdinalEncoder(cols=cat_features, return_df=False).fit(x.copy())
            self.x = enc.transform(self.x.copy())
            print('Data preprocessing ended')
            print('Preprocessed x shape', self.x.shape, 'y shape', self.y.shape)

        if self.resampling != None:
            from resampling import resample_data
            print('RESAMPLING started')
            self.x, self.y = resample_data(self.x.copy(), self.y.copy(),
                                           self.original_num_cols, cat_features,
                                           self.resampling)
            print('RESAMPLING ended')
            print()

        # TODO change
        if self.valtype == 'H':
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(self.x, self.y, test_size=0.2)
            if self.used_algorithms['ELM'] == True:
                self.x_train_ELM, self.x_test_ELM, self.y_train_ELM, \
                self.y_test_ELM = train_test_split(self.x.astype(np.float64),
                                                   self.y_ELM, test_size=0.2)

        def objective_func(args):
            print(args['name'], args['param'])
            model_name = args['name']

            if args['name'] == 'SVM':
                clf = args['model'](
                    kernel=args['param']['kernel'],
                    gamma=args['param']['gamma'],
                    C=args['param']['C'],
                    degree=args['param']['degree']
                )
                if args['scale'] == True:
                    clf = make_pipeline(StandardScaler(), clf)

            elif args['name'] == 'XGBoost':
                import warnings
                warnings.filterwarnings('ignore', message='.*Pass option use_label_encoder=False when')
                clf = args['model']()
                clf.set_params(
                    learning_rate=args['param']['learning_rate'],
                    eval_metric=args['param']['eval_metric']
                )

            elif args['name'] == 'RandomForest':
                clf = args['model'](
                    max_features=args['param']['max_features'],
                    min_samples_leaf=args['param']['min_samples_leaf'],
                    bootstrap=args['param']['bootstrap'])

            elif args['name'] == 'KNeighbors':
                clf = args['model'](
                    n_neighbors=int(args['param']['n_neighbors'])
                )

            elif args['name'] == 'AdaBoost':
                base = args['param']['base_estimator']['model']()
                base.set_params(**args['param']['base_estimator']['param'])
                args['param']['base_estimator'] = base
                clf = args['model']()
                clf.set_params(**args['param'])

            elif args['name'] == 'LinearSVC':
                clf = args['model'](
                    C=args['param']['C'],
                    tol=args['param']['tol'],
                    dual=args['param']['dual'],
                    max_iter=args['param']['max_iter'])
                if args['scale'] == True:
                    clf = make_pipeline(StandardScaler(), clf)

            elif args['name'] == 'HistGB':
                clf = args['model'](
                    learning_rate=args['param']['learning_rate'],
                    max_iter=args['param']['max_iter'],
                    max_depth=args['param']['max_depth'],
                    min_samples_leaf=args['param']['min_samples_leaf'],
                    l2_regularization=args['param']['l2_regularization'],
                )

            elif args['name'] == 'MLP':
                clf = args['model'](
                    hidden_layer_sizes=args['param']['hidden_layer_sizes'],
                    activation=args['param']['activation'],
                    solver=args['param']['solver'],
                    learning_rate=args['param']['learning_rate'],
                    learning_rate_init=args['param']['learning_rate_init'],
                    max_iter=args['param']['max_iter'],
                )
                if args['scale'] is True:
                    clf = make_pipeline(StandardScaler(), clf)

            elif args['name'] == 'LabelSpreading':
                clf = args['model'](
                    kernel=args['param']['kernel'],
                    gamma=args['param']['gamma'],
                    n_neighbors=args['param']['n_neighbors'],
                    alpha=args['param']['alpha'],
                    max_iter=args['param']['max_iter'],
                    tol=args['param']['tol'],
                )

            elif args['name'] == 'LDA':
                clf = args['model'](
                    solver=args['param']['solver'],
                    shrinkage=args['param']['shrinkage'],
                    tol=args['param']['tol'],
                    # priors, n_components, store_covariance are not necessary
                )

            elif args['name'] == 'QDA':
                clf = args['model'](
                    reg_param=args['param']['reg_param'],
                )
            elif args['name'] == 'DecisionTree':
                clf = args['model']()
                clf.set_params(**args['param'])

            elif args['name'] == 'Perceptron':
                clf = args['model']()
                #clf.set_params(**args['param'])

            elif args['name'] == 'GaussianNB':
                clf = args['model']()
                #clf.set_params(**args['param'])

            elif args['name'] == 'Bagging(SVС)':  # rbf
                base = args['param']['base_estimator']['model']()
                base.set_params(**args['param']['base_estimator']['param'])
                args['param']['base_estimator'] = base
                clf = args['model']()
                clf.set_params(**args['param'])
                if args['scale'] == True:
                   clf = make_pipeline(StandardScaler(), clf)

            elif args['name'] == 'xRandTrees':
                clf = args['model']()
                clf.set_params(**args['param'])

            elif args['name'] == 'MetaLearning':
                clf = args['param'][1]()
                clf.set_params(**args['param'][2])
                model_name = model_name + args['param'][0]

            else:
                clf = args['model']()
                # TODO: raise exception
                #raise ValueError('Something wrong with this estimator')
                #clf.set_params(**args['param'])

            if self.valtype == 'CV':
                start_timer = perf_counter()
                if args['name'] == 'ELM':
                    try:
                        cv_results = cross_val_score(clf, self.x_ELM, self.y_ELM, cv=self.kfold,
                                                     scoring=self.metric, n_jobs=self.CV_jobs)
                    except:
                        print("Oops! Error...")
                        cv_results = {}
                        cv_results['memory_fited'] = np.array([9999999999, 9999999999]) # TODO: change to math.inf
                        cv_results['inference_time'] = np.array([9999999999, 9999999999])
                        cv_results['test_score'] = np.array([-9999999999, -9999999999])
                else:
                    cv_results = cross_val_score(clf, self.x, self.y, cv=self.kfold, scoring=self.metric,
                                                 n_jobs=self.CV_jobs)

                mem = cv_results['memory_fited'].max()
                pred_time = cv_results['inference_time'].max()
                accuracy = cv_results['test_score'].mean()
                time_all = perf_counter() - start_timer
            # %%
            elif self.valtype == 'H':
                start_timer = perf_counter()

                if args['name'] == 'ELM':
                    # TODO: add ValueError
                    try:
                        results = split_val_score(clf, self.x_train_ELM, self.x_test_ELM, self.y_train_ELM,
                                                  self.y_test_ELM, scoring=self.metric)
                    except:  # ValueError
                        print("Oops! Error...")
                        results = {}
                        results['memory_fited'] = 9999999999  # TODO: change to math.inf
                        results['inference_time'] = 9999999999
                        results['test_score'] = -9999999999
                else:
                    results = split_val_score(clf, self.x_train, self.x_test, self.y_train, self.y_test,
                                              scoring=self.metric)

                pred_time = results['inference_time']
                mem = results['memory_fited']
                accuracy = results['test_score']
                time_all = perf_counter() - start_timer

            loss = (-accuracy)
            # monitoring
            print(accuracy)
            print('')

            # Model requirements check
            if (accuracy < self.min_accuracy or mem > self.max_model_memory or
                    pred_time > self.max_prediction_time or time_all > self.max_train_time):
                loss = 999
                # TODO: better to change status
                # status = STATUS_FAIL

            status = STATUS_OK

            return {
                'loss': loss,
                'status': status,
                'accuracy': accuracy,
                'model_memory': mem,
                'prediction_time': pred_time,
                'train_time': time_all,
                'model_name': model_name,
                'model': clf
            }

        # Preparing to search
        trials = Trials()
        hyper_space_list = []
        for model in self.models:
            hyper_space_list.append(model.search_space)

        ##################################################
        #              Meta-learning                     #
        ##################################################
        if self.metalearning == True:
            metafeatures = extract_meta_features(self.x, self.y, self.original_cat_cols)
            closest_dids = get_nearest_dids(metafeatures)

            search_space_meta = []
            exceptions = {
                "null": None, 'auto': 'auto', 'scale':'scale',
                'rbf': 'rbf', 'sigmoid': 'sigmoid', 'poly': 'poly', 'linear': 'linear', 'precomputed': 'precomputed',
                'identity': 'identity', 'logistic': 'logistic', 'tanh': 'tanh', 'relu': 'relu',
                'lbfgs': 'lbfgs', 'sgd': 'sgd', 'adam': 'adam',
                'constant': 'constant', 'invscaling': 'invscaling', 'adaptive': 'adaptive',
                'sqrt': 'sqrt',
                'DecisionTreeClassifier':DecisionTreeClassifier,'gini':'gini','entropy':'entropy','log_loss':'log_loss',
                'best':'best','random':'random', 'log2':'log2','balanced':'balanced',
                'ExtraTreeClassifier':ExtraTreeClassifier
            }

            if self.used_algorithms['AdaBoost']:
                print('\nAdaBoost: meta-learning started')
                df_hps = EvalParserAdaBoost.get_optimal_hyperparameters_adaboost(closest_dids, self.mlr_n, 'automl', False)
                df_hps = df_hps.drop(columns=['function', 'value'])
                print(df_hps.to_string())
                for params in df_hps.to_dict('records'):
                    for key in params.keys():
                        params[key] = re.sub(r'min_impurity_split=[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?,', '', params[key])
                        params[key] = re.sub(r'presort=False,', '', params[key],flags=re.IGNORECASE)
                        params[key] = re.sub(r'presort=True,', '', params[key],flags=re.IGNORECASE)
                        # TODO: change it to ast.literal_eval
                        params[key] = eval(params[key], exceptions)
                    search_space_meta.append(['AdaBoost', ensemble.AdaBoostClassifier, params])

            if self.used_algorithms['MLP']:
                print('\nMLP: meta-learning started')
                df_hps = EvalParserMLP.get_optimal_hyperparameters_mlp(closest_dids, self.mlr_n, 'automl', False)
                df_hps = df_hps.drop(columns=['function', 'value'])
                print(df_hps.to_string())
                for params in df_hps.to_dict('records'):
                    for key in params.keys():
                        # TODO: change it to ast.literal_eval
                        params[key] = eval(params[key], exceptions)
                    search_space_meta.append(['MLP', neural_network.MLPClassifier, params])

            if self.used_algorithms['RandomForest']:
                print('\nRandomForest: meta-learning started')
                df_hps = EvalParserRandomForest.get_optimal_hyperparameters_randomforest(closest_dids, self.mlr_n,
                                                                                         'automl', False)
                df_hps = df_hps.drop(columns=['function', 'value'])
                print(df_hps.to_string())
                for params in df_hps.to_dict('records'):
                    for key in params.keys():
                        # TODO: change it to ast.literal_eval
                        params[key] = eval(params[key], exceptions)
                    search_space_meta.append(['RandomForest', ensemble.RandomForestClassifier, params])

            if self.used_algorithms['SVM']:
                print('\nSVM: meta-learning started')
                df_hps = EvalParserSVM.get_optimal_hyperparameters_svm(closest_dids, self.mlr_n, 'automl', False)
                df_hps = df_hps.drop(columns=['function', 'value'])
                print(df_hps.to_string())
                for params in df_hps.to_dict('records'):
                    for key in params.keys():
                        # TODO: change it to ast.literal_eval
                        params[key] = eval(params[key], exceptions)
                    search_space_meta.append(['SVM', svm.SVC, params])

            # TODO: add hp.choice or/and new fmin + Trials that will be reused
            dict_meta = {
                'name': 'MetaLearning',
                'param': hp.choice('Algorithm_Params', search_space_meta)
            }
            hyper_space_list.append(dict_meta)

        # create hyperparameter space
        space = hp.choice('classifier', hyper_space_list)

        # Start search
        try:
            fmin(objective_func, space, algo=tpe.suggest,
                 max_evals=self.iterations, trials=trials, timeout=self.duration)
            self.status = 'OK'
        except hyperopt.exceptions.AllTrialsFailed:
            print('No solutions found. Try a different algorithm or change the requirements')
            self.status = 'no_solution'
        except KeyboardInterrupt:
            print('Execution stopped manually')
            self.status = 'exit'

        if self.status == 'OK':
            # SAVE to EXCEL
            excel_results = []
            for res in trials.results:
                excel_results.append((res['accuracy'], res['model'], res['model_name'], res['model_memory'],
                                      res['prediction_time'], res['train_time']))

            self.results_excel = pd.DataFrame(excel_results,
                                              columns=['accuracy', 'model', 'model_name', 'model_memory',
                                                       'prediction_time', 'train_time'])
            # TODO: add some duplicate filtering
            #self.results_excel.drop_duplicates(subset=['accuracy','model'], inplace=True)

            # save to results trials with only ok status
            results = []
            for res in trials.results:
                if (res['status'] == 'ok') & (res['loss'] < 0):
                    results.append((res['accuracy'], res['model'], res['model_name'], res['model_memory'],
                                    res['prediction_time'], res['train_time']))

            self.optimal_results = results
            self.trials = trials

            # func for sort self.optimal_results
            def sortSecond(val):
                return val[0]

            # sort self.optimal_results by accuracy
            self.optimal_results.sort(key=sortSecond, reverse=True)

            ##################################################
            #               Ensembling                       #
            ##################################################
            if self.ensembling is True:
                estimators = []
                # [( '1_SVM', SVM(p1=..) ), ( '2_RandomForest', RandomForest(p1=..) ), ..., etc]
                for i in range(len(self.optimal_results)):
                    result = self.optimal_results[i]
                    r_name = str(i + 1) + '_' + str(result[2])
                    estimators.append((r_name, result[1]))

                # only use n_estimators from estimators
                if self.n_estimators == 'all':
                    pass
                elif (type(self.n_estimators) == int) & (self.n_estimators > 1):
                    estimators = estimators[:self.n_estimators]
                else:
                    raise Exception("n_estimators - wrong parameter value")

                # create, fit, evaluate VotingClassifier
                clf = VotingClassifier(estimators, voting="hard")
                start_timer = perf_counter()
                if self.valtype == 'CV':
                    cv_results = cross_val_score(clf, self.x, self.y, cv=self.kfold, scoring=self.metric,
                                                 n_jobs=self.CV_jobs)
                    mem = cv_results['memory_fited'].max()
                    pred_time = cv_results['inference_time'].max()
                    accuracy = cv_results['test_score'].mean()
                    time_all = perf_counter() - start_timer
                elif self.valtype == 'H':
                    results = split_val_score(clf, self.x_train, self.x_test, self.y_train, self.y_test,
                                              scoring=self.metric)
                    pred_time = results['inference_time']
                    mem = results['memory_fited']
                    accuracy = results['test_score']
                    time_all = perf_counter() - start_timer
                else:
                    raise Exception("self.valtype - wrong parameter value")
                print('EnsemblingVotingClassifier', accuracy)
                voting_results = (accuracy, clf, 'EnsemblingVotingClassifier', mem, pred_time, time_all)

                # create, fit, evaluate StackingClassifier
                clf = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier()) # , passthrough=True
                start_timer = perf_counter()
                if self.valtype == 'CV':
                    cv_results = cross_val_score(clf, self.x, self.y, cv=self.kfold, scoring=self.metric,
                                                 n_jobs=self.CV_jobs)
                    mem = cv_results['memory_fited'].max()
                    pred_time = cv_results['inference_time'].max()
                    accuracy = cv_results['test_score'].mean()
                    time_all = perf_counter() - start_timer
                elif self.valtype == 'H':
                    results = split_val_score(clf, self.x_train, self.x_test, self.y_train, self.y_test,
                                              scoring=self.metric)
                    pred_time = results['inference_time']
                    mem = results['memory_fited']
                    accuracy = results['test_score']
                    time_all = perf_counter() - start_timer
                else:
                    raise Exception("self.valtype - wrong parameter value")
                print('EnsemblingStackingClassifier', accuracy)
                stacking_results = (accuracy, clf, 'EnsemblingStackingClassifier', mem, pred_time, time_all)

                self.optimal_results.extend([voting_results, stacking_results])
                self.optimal_results.sort(key=sortSecond, reverse=True)

                excel_results.extend([voting_results, stacking_results])
                self.results_excel = pd.DataFrame(excel_results,
                                                  columns=['accuracy', 'model', 'model_name', 'model_memory',
                                                           'prediction_time', 'train_time'])

    def save_results(self, n_best='All', save_excel=True, save_config=True):
        def save_model(to_persist, name):
            dir_name = self.experiment_name
            work_path = os.getcwd()
            path = os.path.join(work_path, dir_name)
            print('Save model: ' + name)
            if os.path.exists(path) == False:
                os.mkdir(path)
            savedir = path
            filename = os.path.join(savedir, name + '.joblib')
            joblib.dump(to_persist, filename)

        # Create folder
        work_path = os.getcwd()
        path = os.path.join(work_path, self.experiment_name)
        if os.path.exists(path) == False:
            os.mkdir(path)

        if n_best == "All":
            for i in range(len(self.optimal_results)):
                model = self.optimal_results[i][1]
                name = str(i + 1) + '_' + str(self.optimal_results[i][2]) + '_' + str(self.optimal_results[i][0])
                save_model(model, name)
        else:
            if isinstance(n_best, int):
                model_num = n_best
            elif n_best == None:
                model_num = None
            elif n_best == "The best":
                model_num = 1
            elif n_best == "Top 5":
                model_num = 5
            elif n_best == "Top 10":
                model_num = 10
            elif n_best == "Top 25":
                model_num = 25
            elif n_best == "Top 50":
                model_num = 50

            if model_num != None:
                if len(self.optimal_results) < model_num:
                    model_num = len(self.optimal_results)
                for i in range(model_num):
                    model = self.optimal_results[i][1]
                    name = str(i + 1) + '_' + str(self.optimal_results[i][2]) + '_' + str(self.optimal_results[i][0])
                    save_model(model, name)

        if save_excel == True:
            self.results_excel.sort_values(by='accuracy', ascending=False, inplace=True)
            self.results_excel.to_excel(self.experiment_name + "\\" + self.experiment_name + "_results.xlsx")

        if save_config == True:
            import config
            cfg = config.default_config.copy()

            # need because when use api you don't have default config.json
            cfg['task'] = 'classification'
            cfg['experiment_name'] = self.experiment_name
            cfg['model_requirements']['min_accuracy'] = self.min_accuracy
            cfg['model_requirements']['max_memory'] = self.max_model_memory
            cfg['model_requirements']['max_single_predict_time'] = self.max_prediction_time
            cfg['model_requirements']['max_train_time'] = self.max_train_time
            cfg['search_space'] = self.used_algorithms
            cfg['search_options']['duration'] = self.duration
            cfg['search_options']['iterations'] = self.iterations
            cfg['search_options']['metric'] = self.metric_original
            cfg['search_options']['validation'] = self.validation
            cfg['search_options']['saved_top_models_amount'] = n_best
            cfg['paths']['DS_abs_path'] = None
            cfg['paths']['CD_abs_path'] = None
            config.save_config(cfg, self.experiment_name + '\\config.json')
            joblib.dump(self.trials, self.experiment_name + '\\hyperopt_trials.pkl')
