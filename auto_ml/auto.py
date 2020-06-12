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
        
        self.__DS=DS
        self.__CD=CD
        self.__experiment_name=experiment_name
        self.__duration=duration
        self.__min_accuracy=min_accuracy
        self.__max_model_memory=max_model_memory
        self.__max_prediction_time=max_prediction_time
        self.__max_train_time=max_train_time
        self.__iterations=iterations
        
        self.__used_algorithms=used_algorithms
        self.__metric=metric
        self.__validation=validation
        
        self.__saved_models_count=saved_models_count
                    
        self.__time_end = perf_counter() + duration
        
        
        
        self.__valtype=''       
        self.__CV_jobs=1              
        self.__cv_splits=None
        
        if(self.__validation in ["3 fold CV","5 fold CV","10 fold CV"]):
            if(self.__validation=="3 fold CV"):
                self.__cv_splits=3
            elif(self.__validation=="5 fold CV"):
                self.__cv_splits=5
            elif(self.__validation=="10 fold CV"):
                self.__cv_splits=10
            self.__valtype='CV'
            from sklearn import model_selection
            self.__kfold = model_selection.KFold(n_splits=self.__cv_splits)

        elif(self.__validation == "holdout"):
            self.__valtype='H'
          
            
                   
        # DEBUG 
        print(self.__DS)
        print(type(self.__DS))
        print(self.__DS.shape)
        print(self.__DS[0])
        print(type(self.__DS[0]))
        
        
        print('!start!')
        preproc = DataPreprocessing(self.__DS,self.__CD)
        self.__x, self.__y = preproc.get_x_y()
        
        self.__y_ELM = preproc.encode_y_ELM_binary(self.__y)
        self.__x_ELM = self.__x.copy()
        self.__x_ELM = self.__x_ELM.astype(np.float64)
       
        self.__nrows, self.__ncol=self.__x.shape  
        
        self.__models=ModelHolder().get_approved_models(self.__used_algorithms)
        
        self.__search()
        
        print('!end!')         

# %%      
    def __check_time(self):           
        if( self.__time_end > perf_counter() ) :
            return True
        else:
            return False           
        
# %%    
    
    def __search(self):     
        from hyperopt import tpe, hp, fmin, STATUS_OK,Trials,STATUS_FAIL
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from util import split_val_score, cross_val_score
                      
        #print(self.__y)
        #print(self.__y_ELM)
        
        # if validation == holdout
        if(self.__valtype == 'H'):
            self.__x_train, self.__x_test, self.__y_train, self.__y_test = \
                                 train_test_split(self.__x,self.__y, test_size=0.2)
            
            if(self.__used_algorithms['ELM']==True):
                self.__x_train_ELM, self.__x_test_ELM, self.__y_train_ELM, \
                  self.__y_test_ELM = train_test_split(self.__x.astype(np.float64),
                                                       self.__y_ELM, test_size=0.2)
        
        #%% 
        def objective_func(args):          
            if(self.__check_time()==True):                    
                
                #debug
                print(args['name'],args['param'])
                
                # every commented parametr worsen performans on G-credit
                # better without them ?
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
                
                #TODO -1 1
                elif args['name']=='ELM': 
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
                                
                if( self.__valtype=='CV' ):                                    
                    start_timer = perf_counter()  
                     
                    if(args['name']=='ELM'):
                        #if ValueError                       
                        try:
                            cv_results = cross_val_score(clf, self.__x_ELM, self.__y_ELM, cv=self.__kfold, scoring = self.__metric, n_jobs = self.__CV_jobs)
                        except :#ValueError
                            print("Oops! Error...") 
                            cv_results={}
                            cv_results['memory_fited']   = np.array([9999999999,9999999999])
                            cv_results['inference_time'] = np.array([9999999999,9999999999])
                            cv_results['test_score']     = np.array([-9999999999,-9999999999])                       
                    else:
                        cv_results = cross_val_score(clf, self.__x, self.__y, cv=self.__kfold, scoring = self.__metric, n_jobs = self.__CV_jobs)
                    
                    mem = cv_results['memory_fited'].max()
                    pred_time = cv_results['inference_time'].max()
                    accuracy = cv_results['test_score'].mean()                                  
                    time_all = perf_counter() - start_timer                   
                #%%               
                elif( self.__valtype == 'H' ):                                
                    start_timer = perf_counter()
                    
                    if(args['name']=='ELM'):
                        #TODO ValueError                       
                        try:
                            results=split_val_score(clf, self.__x_train_ELM, self.__x_test_ELM, self.__y_train_ELM, self.__y_test_ELM, scoring=self.__metric  )
                        except :#ValueError
                            print("Oops! Error...") 
                            results={}
                            results['memory_fited']   = 9999999999
                            results['inference_time'] = 9999999999
                            results['test_score']     = -9999999999                                          
                    else:
                        results=split_val_score(clf, self.__x_train, self.__x_test, self.__y_train, self.__y_test, scoring=self.__metric  )  
                    
                    pred_time = results['inference_time'] 
                    mem = results['memory_fited']  
                    accuracy = results['test_score']           
                    time_all = perf_counter() - start_timer
                #%%               
                loss=(-accuracy)
                
                if(self.__metric=='accuracy'):
                    accuracy=accuracy*100
                
                # monitoring
                print(accuracy)
                print('')
                
                # Model requirments check
                if(accuracy < self.__min_accuracy or 
                   mem > self.__max_model_memory or 
                   pred_time > self.__max_prediction_time or
                   time_all > self.__max_train_time):
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
        for model in self.__models:
            hyper_space_list.append(model.search_space)
                            
        space = hp.choice('classifier',hyper_space_list)
        
        
        # Start search
        import hyperopt        
        
        try:
            fmin(objective_func, space, algo=tpe.suggest, max_evals=self.__iterations, trials=trials)
            self.status='OK'
        except hyperopt.exceptions.AllTrialsFailed:
            print('No solutions found. Try a different algorithm or change the requirements')
            self.status='No solutions found'
        #except:
        #    print('Unexpected error')
        #    self.status='Unexpected error'
            
        
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
            
            self.__optimal_results = results
    
            self.__save_n_best()                
    
# %%    
            
    def __save_n_best(self):
        
        def save_model(to_persist, name):
            import os   
            dir_name=self.__experiment_name    
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
        self.__optimal_results.sort(key = sortSecond, reverse = True)         
        
        
        if(self.__saved_models_count == "Все"):
            for i in range(len(self.__optimal_results)):
                model=self.__optimal_results[i][1]
                name=str(i+1)+'_'+str(self.__optimal_results[i][2])+'_'+str(self.__optimal_results[i][0])
                save_model(model,name)
            
        else:
            if(self.__saved_models_count == "Топ 5"):
                model_num=5
            elif(self.__saved_models_count == "Топ 10"):
                model_num=10
            elif(self.__saved_models_count == "Лучшая"):
                model_num=1
            elif(self.__saved_models_count == "Топ 25"):
                model_num=25
            elif(self.__saved_models_count == "Топ 50"):
                model_num=50
                
            if(len(self.__optimal_results)<model_num):
                model_num = len(self.__optimal_results)   
                        
            for i in range(model_num):
                model=self.__optimal_results[i][1]
                name=str(i+1)+'_'+str(self.__optimal_results[i][2])+'_'+str(self.__optimal_results[i][0])
                save_model(model,name)
        
        self.results_excel.sort_values(by='accuracy', ascending=False,inplace=True)
        self.results_excel.to_excel(self.__experiment_name+"\\model_selection_results.xlsx")
        #TODO save all experiment settings in JSON?
        
#['accuracy']['model']['model_name']['model_memory']['prediction_time']['train_time'] 
   



    
    
    
# %%   #################################################################     
    
        
from category_encoders import OrdinalEncoder
       

class DataPreprocessing:
    
    def __init__(self, DS, CD):
        
        self.__DS = DS.copy()
        self.__CD = CD.copy()
        
        # заменил на очистку при загрузке
        # не работает при category
        # self.__handle_missing()
        self.__col_grouping()       
        
# %%
        
    def __handle_missing(self): # remove row with at least 1 missing value
        self.__DS=self.__DS[np.all(np.isfinite(self.__DS), axis=1)] 
        
# %%         

    def __col_grouping(self):
        
        self.__num_index = []
        self.__categ_index = []
        self.__label_index = None
        
        for column in self.__CD:
            if(column[1]=='Num'):
                self.__num_index.append(column[0]-1)
            elif(column[1]=='Categ'):
                self.__categ_index.append(column[0]-1)  
            elif(column[1]=='Label'):
                self.__label_index = column[0]-1 
                
        self.__num_col   = self.__DS[:,self.__num_index]
        self.__categ_col = self.__DS[:,self.__categ_index]
        self.__label_col = self.__DS[:,self.__label_index]
        
# %%     
    
    def __encode_cat_col(self):
        
        enc = OrdinalEncoder(return_df=False).fit(self.__categ_col)
        self.__categ_col = enc.transform(self.__categ_col) 
        
        # DEBUG
        print(self.__DS)
        print(self.__categ_col)
        # return pandas, IDK why
        # 1TODO pandas to numpy        
     
# %%    
        
    def get_x_y(self):
        # if cat col exist encode
        if(len(self.__categ_index)!=0 ):
            
            self.__encode_cat_col()
            
            if(len(self.__num_index)!=0 ):
                print('has Num, has Categ')
                x = np.hstack([self.__num_col,self.__categ_col])
            else:
                print('no Num, has Categ')
                x=self.__categ_col
            
        else:
            print('no Categ, has Num')
            x=self.__num_col
            
        y=self.__label_col
        
        # x to numpy float x.astype(float)
        return x,y
    
    
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
    
    