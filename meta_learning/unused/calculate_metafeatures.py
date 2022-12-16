import openml
import arff
from pymfe.mfe import MFE
import numpy
import pandas as pd

# v3
def get_features_from_arff(id):
    dataset = openml.datasets.get_dataset(dataset_id=id, download_data=True)
    path2loaded = dataset.data_file
    data = arff.load(open(path2loaded, 'r'))['data']
    # data = dataset.get_data()[0] # doesn't work for some reason
    # features_list = dataset.get_data()[3]
    # if features_list[-1] == : # 'class' 'binaryClass'
    x = [i[:-1] for i in data]
    y = [i[-1] for i in data]
    mfe = MFE(groups="all", random_state=42) # summary="all",
    mfe.fit(x, y)
    ft = mfe.extract(cat_cols='auto', suppress_warnings=True, out_type=pd.DataFrame)
    return ft


# v4
def get_features_from_arff(id):
    dataset = openml.datasets.get_dataset(dataset_id=id, download_data=True)
    X, Y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    X = X.values.tolist()
    Y = Y.tolist()
    categorical_cols = [i for i, x in enumerate(categorical_indicator) if x]

    mfe = MFE(groups="all", random_state=42)  # summary="all",
    mfe.fit(X, Y)
    ft = mfe.extract(cat_cols=categorical_cols, suppress_warnings=True, out_type=pd.DataFrame)
    return ft

