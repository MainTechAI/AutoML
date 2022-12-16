import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from scipy import spatial


def extract_meta_features(ds_X, ds_Y, categorical_cols):
    # define meta-features that you want to compute
    model = MFE(groups="all", random_state=42, measure_time='total_summ')
    allowed_features = set(model.valid_metafeatures())
    excluded_features = ['pb', 'cls_coef', 'density', 'hubs', 'lsc', 'n1', 'n2', 'n3', 'n4', 't1',
                         'conceptvar', 'cohesiveness', 'impconceptvar', 'wg_dist', 'vdu',
                         'l1', 'l2', 'l3', 'two_itemset', 'f1v']
    excluded_features = set(excluded_features)
    allowed_features = allowed_features - excluded_features

    # check if object is numpy.ndarray
    if isinstance(ds_X,np.ndarray) & isinstance(ds_Y,np.ndarray):
        ds_X = pd.DataFrame(data=ds_X)
        ds_Y = pd.DataFrame(data=ds_Y)

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

    #print('categ_num=', len(categorical_cols), 'shape=', ds_X.shape)

    # extract metafeatures
    model = MFE(groups="all", features=allowed_features, random_state=42)
    model = model.fit(X,Y, cat_cols=categorical_cols, precomp_groups=[], suppress_warnings=True, verbose=0)
    res = model.extract(suppress_warnings=True, verbose=1, out_type=pd.DataFrame)  # cat_cols=categorical_cols

    return res.values.tolist()[0]


def get_nearest_dids(metafeatures):
    # The function returns dids of the nearest dids
    # It does it by computing nearest neighbors using KDTree
    # metafeatures: 1D numpy.ndarray of size 107
    dqt_path = r'C:\Users\Trogwald\Desktop\push_it\AutoML\meta_learning\datasets_metafeatures\df_dids_for_train.pkl'
    datasets_qualities_train = pd.read_pickle(dqt_path)
    X_train = datasets_qualities_train.drop('Dataset_ID', axis=1).values
    X_train = np.nan_to_num(X_train)
    metafeatures = np.nan_to_num(metafeatures)

    kd_tree = spatial.KDTree(X_train)
    [d, i] = kd_tree.query(metafeatures, X_train.shape[0])
    i_nearest = pd.Series(i).unique().tolist()
    i_nearest = [x for x in i_nearest if x != 492] # 492 is out of boundary value

    # now find those dids by their indexes
    a = datasets_qualities_train.Dataset_ID.values
    dids_predicted = np.take(a, i_nearest)
    dids_predicted = list(map(int, dids_predicted))

    # just want to be sure that all train_dids are used
    dids_train = datasets_qualities_train.Dataset_ID.to_list()
    dids_train = set(list(map(int, dids_train)))
    if len(dids_train - set(dids_predicted)) == 0:
        return dids_predicted
    else:
        diff_dids = list( dids_train - set(dids_predicted) )
        dids_predicted.extend(diff_dids)
        return dids_predicted


def clear_strings_in_dict(parsed):
    for key,val in parsed.items():
        if isinstance(val, str):
            parsed[key] = parsed[key].replace('\"', '').replace('\'', '')
            if val.lower() == 'true':
                parsed[key] = 'True'
            if val.lower() == 'false':
                parsed[key] = 'False'
            parsed[key] = " ".join(parsed[key].split())

    return parsed

