import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from AutoML.meta_learning.util.utils import get_nearest_dids, clear_strings_in_dict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_optimal_hyperparameters_randomforest(closest_dids, n=1, p_type='automl', verbose=True):
    # the function obtains the best hyperparameters according to the most similar datasets
    # the most relevant results will be obtained for
    df_results = pd.DataFrame()
    for dataset_id in closest_dids:
        df_res_did = get_parameters_by_did_randomforest(dataset_id, n=n, p_type=p_type)
        if df_res_did.shape[0] != 0:
            df_results = df_results.append(df_res_did, ignore_index=True)
            df_results = df_results.drop_duplicates(subset=df_results.columns.difference(['function','value']))
        if df_results.shape[0] >= n:
            break
        if verbose == True:
            print(f'dataset[id='+str(dataset_id)+'] processed, '+str(df_results.shape[0])+'/'+str(n)+' runs obtained')
    return df_results.iloc[:n]


def get_parameters_by_did_randomforest(did, n=1, p_type='automl'):
    # n: int(0>n) or str('all') - number of results returned. By default, only single best result is returned.
    # p_type: str = 'automl' return only parameters relevant to the AutoML app (small subset)
    #         str = 'all' return default parameters (the whole set), use in case if there is not enough results
    #
    # returns parameters of the best evaluations from OpenML
    #         when there is no such results, returns False

    res_path = 'C:\\Users\\Trogwald\\Desktop\\push_it\\AutoML\\meta_learning\\\parameters_results_study1\\'+str(did)+'\\'

    pkls = [f for f in listdir(res_path) if isfile(join(res_path, f))]
    tids = [int(f.replace('.pkl','')) for f in pkls]

    df_results = pd.DataFrame()
    for tid in tids:
        # combine results from all task_ids
        df_res_tid = get_params_randomforest(did, tid, n, p_type)
        if df_res_tid.shape[0] != 0:
            df_results = df_results.append(df_res_tid, ignore_index=True)
            df_results = df_results.drop_duplicates(subset=df_results.columns.difference(['function','value']))

    # get only n best results
    if df_results.shape[0] != 0:
        df_results.sort_values(by='value', ascending=False, inplace=True)

    results = df_results.iloc[:n]
    return results


# The most important function I guess
def get_params_randomforest(did, tid, num_results, p_type='automl'):
    # Load all evaluations for a given task_id
    path = 'C:\\Users\\Trogwald\\Desktop\\push_it\\AutoML\\meta_learning\\'
    fpath = path + 'parameters_results_study1\\' + str(did) + '\\' + str(tid) + '.pkl'
    df = pd.read_pickle(fpath)

    # Filter out other packages (Weka, R, etc)
    FLOW_IDS_ALLOWED = pd.read_pickle(path+r'util\df_sklearn_flows.pkl').id.to_list()
    df = df.loc[df['flow_id'].isin(FLOW_IDS_ALLOWED)].reset_index(drop=True)

    # 1. Strict search for RandomForest algorithm (without Pipeline, VotingClassifier, etc)
    df_randomforest = df[
        df['flow_name'].str.contains('^sklearn.ensemble.forest.RandomForestClassifier',regex=True,case=False) |
        df['flow_name'].str.contains('^sklearn.ensemble._forest.RandomForestClassifier',regex=True,case=False)
    ]  # TODO I didn't properly tested this but the second condition should be useful
    df_randomforest.reset_index(drop=True)
    num_rows = df_randomforest.shape[0]

    df_params = pd.DataFrame()
    if num_rows != 0:
        for metric,value,param_dict in df_randomforest[["function", "value", "parameters"]].values:
            # I don't need duplicated parameters, iterate until rows != num_results
            if p_type == 'automl':
                parsed = parse_randomforest(param_dict)
            elif p_type == 'all':
                parsed = parse_randomforest_all(param_dict)
            #TODO not sure about this thing
            if type(parsed) == bool:
                #print('False',param_dict)
                continue
            parsed['function'] = metric
            parsed['value'] = value
            df_params = df_params.append(parsed, ignore_index=True)
            df_params = df_params.drop_duplicates(subset=df_params.columns.difference(['function','value']))
            if df_params.shape[0] == num_results:
                break

    # you can implement less strict rules in case if df_params.shape[0] < num_results
    # ...
    return df_params


def parse_randomforest(param_dict):
    hp_my = ['max_features','min_samples_leaf','bootstrap']
    all_hp_dict = {}
    for hp in hp_my:
        hp_dict = {hp:val for key, val in param_dict.items() if key.endswith(hp)}
        if hp_dict:
            all_hp_dict = {**all_hp_dict, **hp_dict}
        else:
            return False
    return clear_strings_in_dict(all_hp_dict)


def parse_randomforest_all(param_dict):
    # Attention
    # The parameters are inconsistent across all OpenML evals.
    # You need to handle missing keys
    hp_default = ["n_estimators","criterion","max_depth","min_samples_split","min_samples_leaf","min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease","min_impurity_split","bootstrap","oob_score","n_jobs","random_state","class_weight","ccp_alpha","max_samples"]
    all_hp_dict = {}

    for hp in hp_default:
        hp_dict = {hp:val for key, val in param_dict.items() if key.endswith(hp)}
        if hp_dict:
            all_hp_dict = {**all_hp_dict, **hp_dict}
        #else:
        #    return False
    return clear_strings_in_dict(all_hp_dict)


if __name__ == "__main__":
    # Test data (20 samples from OpenML)
    dqt = r'C:\Users\Trogwald\Desktop\push_it\AutoML\meta_learning\datasets_metafeatures\df_dids_for_test.pkl'
    datasets_qualities_test = pd.read_pickle(dqt)
    X_test = datasets_qualities_test.drop('Dataset_ID', axis=1).values
    metafeatures = X_test[1]

    n = 10
    p_type = 'automl'  # options 'automl, 'all'
    closest_dids = get_nearest_dids(metafeatures)

    df_hps = get_optimal_hyperparameters_randomforest(closest_dids, n, p_type)
    print(df_hps.to_string())
