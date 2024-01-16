# this script computes meta-features for all datasets in a given Study(collection of datasets)
# and then, preprocesses them
import openml
import pandas as pd
from pymfe.mfe import MFE
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# TODO: change absolute paths to relative ones


study_id = 1  # ~500 instances of classification datasets

openml.config.apikey = ''  # insert your API key here
openml.config.retry_policy = 'human'
openml.config.connection_n_retries = 50


# get list of allowed features
model = MFE(groups="all",random_state=42, measure_time='total_summ')
allowed_features = set(model.valid_metafeatures())
excluded_features = ['pb','cls_coef','density', 'hubs','lsc','n1','n2','n3','n4','t1',
                     'conceptvar','cohesiveness', 'impconceptvar','wg_dist', 'vdu',
                     'l1', 'l2', 'l3','two_itemset','f1v']
excluded_features = set(excluded_features)
allowed_features = allowed_features - excluded_features


def extract_meta_features_openml(dataset_id, save_dir):
    # get dataset
    dataset = openml.datasets.get_dataset(dataset_id=dataset_id, download_data=True)
    ds_X, ds_Y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    ds_X['RESULT_LABEL_COLUMN5184_Y'] = ds_Y.values

    # drop duplicate values
    ds_X = ds_X.drop_duplicates()

    # Convert nan to category NaN, 0 for the rest
    cat_cols = ds_X.select_dtypes(include='category').columns.tolist()
    for cat_name in cat_cols:
        new_cat = "?"
        if new_cat not in ds_X[cat_name].cat.categories.tolist():
            ds_X[cat_name].cat.add_categories(new_cat, inplace=True)
            ds_X[cat_name].fillna(new_cat, inplace=True)
        else:
            ds_X[cat_name].fillna(new_cat, inplace=True)
    ds_X.fillna(0, inplace=True)

    # sample data
    eff_num_samples = ds_X.shape[1] * 10
    if eff_num_samples < ds_X.shape[0]:
        ds_X = ds_X.sample(n=eff_num_samples, random_state=42, replace=False)
    if 1000 < ds_X.shape[0]:
        ds_X = ds_X.sample(n=1000, random_state=42, replace=False)

    # preprocessing
    Y = ds_X['RESULT_LABEL_COLUMN5184_Y'].values.tolist()
    ds_X = ds_X.drop('RESULT_LABEL_COLUMN5184_Y', axis=1)
    X = ds_X.values.tolist()

    categorical_cols = [i for i, x in enumerate(categorical_indicator) if x]
    print(dataset_id, ':', 'categ_num=', len(categorical_cols), 'shape=', ds_X.shape)

    # extract metafeatures
    model = MFE(groups="all", features=allowed_features, random_state=42)
    model = model.fit(X,Y, cat_cols=categorical_cols, precomp_groups=[], suppress_warnings=True, verbose=1)
    res = model.extract(suppress_warnings=True, verbose=2, out_type=pd.DataFrame)  # cat_cols=categorical_cols

    # save results
    path = save_dir + str(dataset_id) + '.pkl'
    res.to_pickle(path)

    print("Meta-features of Dataset(id=" + str(dataset_id) + ') were extracted')


# datasets_id from Study1 (~500 datasets)
study = openml.study.get_study(study_id=study_id)
print(study.data)
folder = r"C:\Users\Trogwald\Desktop\push_it\AutoML\meta_learning\datasets_metafeatures\\"


for ds_id in study.data:
    extract_meta_features_openml(ds_id, folder)


#######################################
# Combine results into a single csv file
#######################################

study1_metafeatures = pd.DataFrame()

for ds_id in study.data:
    path = folder + str(ds_id) + '.pkl'
    loaded_mf = pd.read_pickle(path)
    loaded_mf['Dataset_ID'] = ds_id
    study1_metafeatures = study1_metafeatures.append(loaded_mf, ignore_index=True)

column1 = study1_metafeatures.pop('Dataset_ID')
study1_metafeatures.insert(0, 'Dataset_ID', column1)

study1_metafeatures.to_csv(folder+'study1_metafeatures.csv',index=False)


#######################################
# the resulting file contains all datasets from Study11
# but some of the datasets need to be removed, as there are no TIDs for them
#######################################


folder = r"C:\Users\Trogwald\Desktop\push_it\AutoML\meta_learning\\"


all_dids = pd.read_csv(folder+'datasets_metafeatures\\study1_metafeatures.csv').Dataset_ID.values.tolist()
all_dids = set(all_dids)

did_tid = pd.read_pickle(folder+'util\\did_tid.pkl')
task_exist = []
task_not_exist = []
dataset_exist = []


for did, tid_list in did_tid.items():
    for tid in tid_list:
        file_path = folder + 'parameters_results_study1\\' + str(did) + '\\' + str(tid) + '.pkl'

        is_exist = os.path.exists(file_path)
        if is_exist:
            task_exist.append(tid)
            dataset_exist.append(did)
        else:
            task_not_exist.append(tid)

# remove duplicates
dataset_exist = set(dataset_exist)

# datasets from Study1 that were excluded
dataset_not_exist = all_dids - dataset_exist

print('Found: '+str(len(task_exist))+' tasks')
print('Not found: '+str(len(task_not_exist))+' tasks')
print()
print('Found: '+str(len(dataset_exist))+' datasets')
print('Not found: '+str(len(dataset_not_exist))+' datasets')

dataset_exist = list(dataset_exist)

dataset_not_exist = list(dataset_not_exist)
dataset_not_exist = list(map(int, dataset_not_exist))

# Split dataset with meta-features into two datasets (train, test)
all_meta_features = pd.read_csv(folder+'datasets_metafeatures\\study1_metafeatures.csv')

df_dids_for_train = all_meta_features.loc[all_meta_features['Dataset_ID'].isin(dataset_exist)].reset_index(drop=True)
df_dids_for_train.to_pickle(folder+'datasets_metafeatures\\df_dids_for_train.pkl')

df_dids_for_test = all_meta_features.loc[all_meta_features['Dataset_ID'].isin(dataset_not_exist)].reset_index(drop=True)
df_dids_for_test.to_pickle(folder+'datasets_metafeatures\\df_dids_for_test.pkl')

# 20 datasets for test
# 492 datasets for train
