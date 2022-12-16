# this script downloads (not calculates) all meta-features from all datasets in a given study (collection of datasets)
# the script is not used
import openml
import numpy
import pandas as pd
import json


study_id = 1  # ~500 instances of classification datasets

openml.config.apikey = ''  # insert your API key here
openml.config.retry_policy = 'robot'
openml.config.connection_n_retries = 50


def save_meta_features(dataset_id, save_dir):
    dataset = openml.datasets.get_dataset(dataset_id=dataset_id, download_data=False, download_qualities=True)
    path = save_dir + str(dataset_id) + '.json'
    with open(path, 'w') as f:
        json.dump(dataset.qualities, f)
    print("Qualities of Dataset(id=" + str(dataset_id) + ') were saved')


# load MetaFeatures from Study1 (~500 datasets)
study = openml.study.get_study(study_id=study_id)
print(study.data)
folder = r"C:\Users\Trogwald\Desktop\push_it\AutoML\meta_learning\datasets_study1\\"
for ds_id in study.data:
    save_meta_features(ds_id,folder)


#######################################
# Combine results into a single csv file
#######################################

study1_metafeatures = pd.DataFrame()

for ds_id in study.data:
    path = folder + str(ds_id) + '.json'
    with open(path, 'r') as fp:
        loaded_mf = json.load(fp)
        loaded_mf['Dataset_ID'] = ds_id
        study1_metafeatures = study1_metafeatures.append(loaded_mf, ignore_index=True)

column1 = study1_metafeatures.pop('Dataset_ID')
study1_metafeatures.insert(0, 'Dataset_ID', column1)

study1_metafeatures.to_csv(folder+'study1_metafeatures.csv',index=False)

# the resulting file contains all dataset from Study1
# but some of the datasets were removed later, as they don't fit the
