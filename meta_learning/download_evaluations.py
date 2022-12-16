# The script downloads all evaluations_setups for task_ids from study1
# The runtime was about one day
import openml
import pandas as pd
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

openml.config.apikey = ''  # insert your apikey
openml.config.retry_policy = 'robot'
openml.config.connection_n_retries = 150

path = r"C:\Users\Trogwald\Desktop\push_it\AutoML\meta_learning\\"
folder = r'parameters_results_study1\\'

did_tid = pd.read_pickle(path+r'util\did_tid.pkl')


for did, tid_list in did_tid.items():
    print('did['+str(did)+']')
    for tid in tid_list:
        print('  tid['+str(tid)+']')
        fpath = path + folder + str(did) + '\\'

        evals = openml.evaluations.list_evaluations_setups(
            function='predictive_accuracy',
            tasks=[tid],
            sort_order = 'desc',output_format = 'dataframe')

        if evals.empty == False:
            Path(path + folder + str(did) + '\\').mkdir(parents=True, exist_ok=True)
            evals.to_pickle(fpath+str(tid)+'.pkl')