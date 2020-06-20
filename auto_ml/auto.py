# -*- coding: utf-8 -*-

from time import perf_counter
import numpy as np
from models import ModelHolder

import pandas as pd


class ModelSelection: 
                     
    def __init__(self, DS, CD, experiment_name, duration, min_accuracy,
                 max_model_memory, max_prediction_time, max_train_time, 
                 used_algorithms, metric, validation, saved_models_count,
                 iterations):
        
        ###!!!  DEV
        
        self.row_count = None                    
        self.columns_count = None # all col (with target?)
        self.target_column = None
        self.cat_columns = []
        
        ###!!!  DEV
        
        self.DS=DS
        self.CD=CD
        self.experiment_name=experiment_name
        self.duration=duration
        self.min_accuracy=min_accuracy
        self.max_model_memory=max_model_memory
        self.max_prediction_time=max_prediction_time
        self.max_train_time=max_train_time
        self.iterations=iterations
        
        self.used_algorithms=used_algorithms
        self.metric=metric
        self.validation=validation
        
        self.saved_models_count=saved_models_count
                    
        self.time_end = perf_counter() + duration
        
        
        
        self.valtype=''       
        self.CV_jobs=1              
        self.cv_splits=None
        
        if(self.validation in ["3 fold CV","5 fold CV","10 fold CV"]):
            if(self.validation=="3 fold CV"):
                self.cv_splits=3
            elif(self.validation=="5 fold CV"):
                self.cv_splits=5
            elif(self.validation=="10 fold CV"):
                self.cv_splits=10
            self.valtype='CV'
            from sklearn import model_selection
            self.kfold = model_selection.KFold(n_splits=self.cv_splits)

        elif(self.validation == "holdout"):
            self.valtype='H'
          
            
                   
        # DEBUG 
        print(self.DS)
        print(type(self.DS))
        print(self.DS.shape)
        print(self.DS[0])
        print(type(self.DS[0]))
        
        
        print('!start!')
        preproc = DataPreprocessing(self.DS,self.CD)
        self.x, self.y = preproc.get_x_y()
        
        self.y_ELM = preproc.encode_y_ELM_binary(self.y)
        self.x_ELM = self.x.copy()
        self.x_ELM = self.x_ELM.astype(np.float64)
       
        self.nrows, self.ncol=self.x.shape  
        
        self.models=ModelHolder().get_approved_models(self.used_algorithms)
        
        self.search()
        
        print('!end!')         

# %%      
    def check_time(self):           
        if( self.time_end > perf_counter() ) :
            return True
        else:
            return False           
        
# %%    
    
    def search(self):     
        from hyperopt import tpe, hp, fmin, STATUS_OK,Trials,STATUS_FAIL
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from util import split_val_score, cross_val_score
                      
        #print(self.y)
        #print(self.y_ELM)
        
        # if validation == holdout
        if(self.valtype == 'H'):
            self.x_train, self.x_test, self.y_train, self.y_test = \
                                 train_test_split(self.x,self.y, test_size=0.2)
            
            if(self.used_algorithms['ELM']==True):
                self.x_train_ELM, self.x_test_ELM, self.y_train_ELM, \
                  self.y_test_ELM = train_test_split(self.x.astype(np.float64),
                                                       self.y_ELM, test_size=0.2)
        
        #%% 
        def objective_func(args):          
            if(self.check_time()==True):                    
                
                #debug
                print(args['name'],args['param'])
                
                # every commented parametr worsen performans on G-credit
                # better without them 
                if args['name']=='SVM':             
                    clf = args['model'](
                        kernel = args['param']['kernel'],
                        gamma = args['param']['gamma'],
                        C = args['param']['C'],
                        degree = args['param']['degree']
                    )
                    if(args['scale']==True):                
                        clf = make_pipeline(StandardScaler(), clf)

                elif args['name']=='XGBoost':                     
                    clf = args['model'](
                        learning_rate = args['param']['learning_rate'],                     
                        #  убрал из-за низкой эффективности
                        #booster = args['param']['booster'],
                        #n_estimators = args['param']['n_estimators'],                          
                        #subsample = args['param']['subsample'],
                        #max_depth = args['param']['max_depth'],
                        #min_child_weight = args['param']['min_child_weight'],
                        #colsample_bytree = args['param']['colsample_bytree'],
                        #colsample_bylevel = args['param']['colsample_bylevel'],
                        #reg_lambda = args['param']['reg_lambda']  ,      
                        #reg_alpha = args['param']['reg_alpha']  ,              
                    )
                    # scale не нужно 
                        
                elif args['name']=='RandomForest': 
                    clf = args['model'](
                        max_features =  args['param']['max_features'],
                        min_samples_leaf =  args['param']['min_samples_leaf'],
                        bootstrap =  args['param']['bootstrap'])
                    
                elif args['name']=='KNeighbors': 
                    clf = args['model'](
                        n_neighbors = int(args['param']['n_neighbors'])
                        )
                    
                elif args['name']=='AdaBoost':
                    if(args['param']['base_estimator']['name']=='DecisionTree'):
                        base=args['param']['base_estimator']['model'](
                                max_depth = args['param']['base_estimator']['max_depth'])                   
                    clf = args['model'](
                        learning_rate =  args['param']['learning_rate'],
                        base_estimator = base)    
                    
                elif args['name']=='LinearSVC': 
                    clf = args['model'](
                        C =  args['param']['C'],
                        tol =  args['param']['tol'],
                        dual =  args['param']['dual'],
                        max_iter =  args['param']['max_iter'])
                    if(args['scale']==True):               
                        clf = make_pipeline(StandardScaler(), clf)
                 
                elif args['name']=='HistGB': 
                    clf = args['model'](
                        learning_rate = args['param']['learning_rate'],
                        #max_iter =  args['param']['max_iter'],
                        #max_depth = args['param']['max_depth'],
                        #min_samples_leaf = args['param']['min_samples_leaf'],
                        #l2_regularization = args['param']['l2_regularization'],
                        )      
                    
                elif args['name']=='MLP':
                    clf = args['model'](
                        hidden_layer_sizes =  args['param']['hidden_layer_sizes'],
                        activation =  args['param']['activation'],
                        solver = args['param']['solver'],
                        learning_rate = args['param']['learning_rate'],
                        learning_rate_init = args['param']['learning_rate_init'],
                        max_iter = args['param']['max_iter'],
                        )  
                    
                    if(args['scale']==True):                
                        clf = make_pipeline(StandardScaler(), clf)
                        
                elif args['name']=='LabelSpreading': 
                    clf = args['model'](
                        kernel = args['param']['kernel'],
                        gamma = args['param']['gamma'],
                        n_neighbors =  args['param']['n_neighbors'],
                        alpha =  args['param']['alpha'],
                        max_iter =  args['param']['max_iter'],
                        tol =  args['param']['tol'],
                        )          
                
                elif args['name']=='LDA': 
                    clf = args['model'](
                        solver = args['param']['solver'],
                        shrinkage = args['param']['shrinkage'],
                        tol = args['param']['tol'],
                        #priors, n_components, store_covariance не нужены
                        ) 
                    
                elif args['name']=='QDA': 
                    clf = args['model'](
                        reg_param = args['param']['reg_param'],
                        )                                 
                
                elif args['name']=='ELM':
                    #TODO -1 1
                    clf = args['model'](
                        hid_num = int(args['param']['hid_num']),
                        a =  args['param']['a'],
                        )
                
                elif args['name']=='Bagging(SVС)': # rbf
                    base=args['param']['base_estimator']['model'](
                        kernel = args['param']['base_estimator']['kernel'],
                        gamma = args['param']['base_estimator']['gamma'],
                        C = args['param']['base_estimator']['C'],                   
                    )                  
                    clf = args['model'](
                        base_estimator = base,                        
                        n_estimators = args['param']['n_estimators'],
                        )                  
                    if(args['scale']==True):                
                        clf = make_pipeline(StandardScaler(), clf)
                

                
                                                  
                else:
                    clf = args['model']()
                    # TODO add other                
                
                #%%
                                
                if( self.valtype=='CV' ):                                    
                    start_timer = perf_counter()  
                     
                    if(args['name']=='ELM'):
                        #if ValueError                       
                        try:
                            cv_results = cross_val_score(clf, self.x_ELM, self.y_ELM, cv=self.kfold, scoring = self.metric, n_jobs = self.CV_jobs)
                        except :#ValueError
                            print("Oops! Error...") 
                            cv_results={}
                            cv_results['memory_fited']   = np.array([9999999999,9999999999])
                            cv_results['inference_time'] = np.array([9999999999,9999999999])
                            cv_results['test_score']     = np.array([-9999999999,-9999999999])                       
                    else:
                        cv_results = cross_val_score(clf, self.x, self.y, cv=self.kfold, scoring = self.metric, n_jobs = self.CV_jobs)
                    
                    mem = cv_results['memory_fited'].max()
                    pred_time = cv_results['inference_time'].max()
                    accuracy = cv_results['test_score'].mean()                                  
                    time_all = perf_counter() - start_timer                   
                #%%               
                elif( self.valtype == 'H' ):                                
                    start_timer = perf_counter()
                    
                    if(args['name']=='ELM'):
                        #TODO ValueError                       
                        try:
                            results=split_val_score(clf, self.x_train_ELM, self.x_test_ELM, self.y_train_ELM, self.y_test_ELM, scoring=self.metric  )
                        except :#ValueError
                            print("Oops! Error...") 
                            results={}
                            results['memory_fited']   = 9999999999
                            results['inference_time'] = 9999999999
                            results['test_score']     = -9999999999                                          
                    else:
                        results=split_val_score(clf, self.x_train, self.x_test, self.y_train, self.y_test, scoring=self.metric  )  
                    
                    pred_time = results['inference_time'] 
                    mem = results['memory_fited']  
                    accuracy = results['test_score']           
                    time_all = perf_counter() - start_timer
                #%%               
                loss=(-accuracy)
                
                if(self.metric=='accuracy'):
                    accuracy=accuracy*100
                
                # monitoring
                print(accuracy)
                print('')
                
                # Model requirments check
                if(accuracy < self.min_accuracy or 
                   mem > self.max_model_memory or 
                   pred_time > self.max_prediction_time or
                   time_all > self.max_train_time):
                    status=STATUS_FAIL
                    loss=999
                else:
                    status=STATUS_OK
                
                return {
                        'loss':loss,
                        'status': status,
                        'accuracy': accuracy,
                        'model_memory': mem,
                        'prediction_time': pred_time,
                        'train_time': time_all,
                        'model_name':args['name'],
                        'model':clf
                        }
            else:                
                return {
                        'loss':None,
                        'status': STATUS_FAIL,
                        'accuracy': None,
                        'model_memory': None,
                        'prediction_time': None,
                        'train_time': None,
                        'model_name': None,
                        'model':None
                        }
        
        #%%
        
        # Prepairing to search      
        trials = Trials()      
        hyper_space_list=[]
        for model in self.models:
            hyper_space_list.append(model.search_space)
                            
        space = hp.choice('classifier',hyper_space_list)
        
        
        # Start search
        import hyperopt        
        
        try:
            fmin(objective_func, space, algo=tpe.suggest, max_evals=self.iterations, trials=trials)
            self.status='OK'
        except hyperopt.exceptions.AllTrialsFailed:
            print('No solutions found. Try a different algorithm or change the requirements')
            self.status='No solutions found'
        except:
            print('Unexpected error')
            self.status='Unexpected error'
            
        
        #%%    
        if(self.status=='OK'):
            # SAVE to EXCEL
            excel_results=[]
            for res in trials.results:
                excel_results.append( (res['accuracy'],res['model'],res['model_name'],res['model_memory'],res['prediction_time'],res['train_time']) )
            
            self.results_excel = pd.DataFrame( excel_results,  
                                               columns = ['accuracy','model','model_name','model_memory','prediction_time','train_time'] ) 
            
            # save results with only ok status      
            results=[]
            for res in trials.results:
                if( res['status']=='ok'):
                    results.append( (res['accuracy'],res['model'],res['model_name'],res['model_memory'],res['prediction_time'],res['train_time']) )     
            
            self.optimal_results = results
    
            self.save_n_best()                
    
# %%    
            
    def save_n_best(self):
        
        def save_model(to_persist, name):
            import os   
            dir_name=self.experiment_name    
            work_path = os.getcwd()
            path = os.path.join(work_path, dir_name) 
            print('Save model: '+name)
            if(os.path.exists(path)==False):
                os.mkdir(path)          
            savedir = path
            filename = os.path.join(savedir, name+'.joblib')           
            import joblib
            joblib.dump(to_persist, filename) 
        
        
        # func for sort self.optimal_results
        def sortSecond(val): 
        	return val[0] 
        
        
        # sort self.optimal_results by accuracy
        self.optimal_results.sort(key = sortSecond, reverse = True)         
        
        
        if(self.saved_models_count == "Все"):
            for i in range(len(self.optimal_results)):
                model=self.optimal_results[i][1]
                name=str(i+1)+'_'+str(self.optimal_results[i][2])+'_'+str(self.optimal_results[i][0])
                save_model(model,name)
            
        else:
            if(self.saved_models_count == "Топ 5"):
                model_num=5
            elif(self.saved_models_count == "Топ 10"):
                model_num=10
            elif(self.saved_models_count == "Лучшая"):
                model_num=1
            elif(self.saved_models_count == "Топ 25"):
                model_num=25
            elif(self.saved_models_count == "Топ 50"):
                model_num=50
                
            if(len(self.optimal_results)<model_num):
                model_num = len(self.optimal_results)   
                        
            for i in range(model_num):
                model=self.optimal_results[i][1]
                name=str(i+1)+'_'+str(self.optimal_results[i][2])+'_'+str(self.optimal_results[i][0])
                save_model(model,name)
        
        self.results_excel.sort_values(by='accuracy', ascending=False,inplace=True)
        self.results_excel.to_excel(self.experiment_name+"\\model_selection_results.xlsx")
        #TODO save all experiment settings in JSON?
        
#['accuracy']['model']['model_name']['model_memory']['prediction_time']['train_time'] 
   



    
# %%   #################################################################   
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# %% 
        
from category_encoders import OrdinalEncoder
       

class DataPreprocessing:
    
    def __init__(self, DS, CD):
        
        self.DS = DS.copy()
        self.CD = CD.copy()
        
        # заменил на очистку при загрузке
        # не работает при category
        # self.handle_missing()
        self.col_grouping()       
        
# %%
        
    def handle_missing(self): # remove row with at least 1 missing value
        self.DS=self.DS[np.all(np.isfinite(self.DS), axis=1)] 
        
# %%         

    def col_grouping(self):
        
        self.num_index = []
        self.categ_index = []
        self.label_index = None
        
        for column in self.CD:
            if(column[1]=='Num'):
                self.num_index.append(column[0]-1)
            elif(column[1]=='Categ'):
                self.categ_index.append(column[0]-1)  
            elif(column[1]=='Label'):
                self.label_index = column[0]-1 
                
        self.num_col   = self.DS[:,self.num_index]
        self.categ_col = self.DS[:,self.categ_index]
        self.label_col = self.DS[:,self.label_index]
        
# %%     
    
    def encode_cat_col(self):
        
        enc = OrdinalEncoder(return_df=False).fit(self.categ_col)
        self.categ_col = enc.transform(self.categ_col) 
        
        # DEBUG
        print(self.DS)
        print(self.categ_col)
        # return pandas, IDK why
        # 1TODO pandas to numpy        
     
# %%    
        
    def get_x_y(self):
        # if cat col exist encode
        if(len(self.categ_index)!=0 ):
            
            self.encode_cat_col()
            
            if(len(self.num_index)!=0 ):
                print('has Num, has Categ')
                x = np.hstack([self.num_col,self.categ_col])
            else:
                print('no Num, has Categ')
                x=self.categ_col
            
        else:
            print('no Categ, has Num')
            x=self.num_col
            
        y=self.label_col
        
        # x to numpy float x.astype(float)
        #.astype(float)
        return x.astype(float),y
    
    
# %%
        
    # данная реализация ELM требует на вход 1 и -1
    def encode_y_ELM_binary(self,y_input):
        y=y_input.copy()
        for i in range(len(y)):
            if(y[i]==y[0]):
                y[i]=1
            else:
                y[i]=-1
        return y.astype(np.int8)
    
# %%    
    
    
    
    
    
    
    
    
    
    
# %%
        
#from hyperopt import fmin, tpe, hp, STATUS_OK
#
#def objective(x):
#    return {'loss': x ** 2, 'status': STATUS_OK }
#
#best = fmin(objective,
#    space=hp.uniform('x', -10, 10),
#    algo=tpe.suggest,
#    max_evals=1000)
#
#print( best )




#
#from model_list import ClassificationModels
#
#model_list = ClassificationModels().get_approved_models()
#print(model_list)
#
#
#
#import pickle
#import time
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#
#print(model_list[1])
#
##def objective(model_list[1].default_parameters):
#    return {
#        'loss': model_list[1].get_skl_estimator(),
#        'status': STATUS_OK,
#        # -- store other results like this
#        'eval_time': time.time(),
#        'other_stuff': {'type': None, 'value': [0, 1, 2]},
#        # -- attachments are handled differently
#        'attachments':
#            {'time_module': pickle.dumps(time.time)}
#        }
#        
#trials = Trials()
#best = fmin(objective,
#    space=hp.uniform('x', -10, 10),
#    algo=tpe.suggest,
#    max_evals=100,
#    trials=trials)
#
#print(best)        
    







def foo(): # всё ок

    import os
    
    dir_name='experiment2'
    
    work_path = os.getcwd() # current working dir
    path = os.path.join(work_path, dir_name) 
    print ("The current working directory is %s" % work_path)
    
    if(os.path.exists(path)==False):
        os.mkdir(path)
    else:
        print('Directory already exist')
    
    
    savedir = path
    import os
    filename = os.path.join(savedir, 'model.joblib')
        
    
    from sklearn.datasets import load_breast_cancer
    X, Y = load_breast_cancer(return_X_y=True)
    
    #from sklearn.gaussian_process import GaussianProcessClassifier
    #to_persist=GaussianProcessClassifier()
    
    #from lightning.classification import AdaGradClassifier
    #to_persist=AdaGradClassifier()
    
    from dbn import SupervisedDBNClassification
    to_persist=SupervisedDBNClassification()
     
    
    
    to_persist.fit(X[:400],Y[:400])
    
    print(filename)
    
    import joblib
    joblib.dump(to_persist, filename) 
    
    
    # load from file
    import joblib
    clf = joblib.load(filename) 
    
    print(clf.score(X[400:],Y[400:]))



    