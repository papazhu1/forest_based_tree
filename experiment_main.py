import pandas as pd
import numpy as np
from ExperimentSetting import *
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import glob
warnings.filterwarnings('ignore')
import time

branch_probability_thresholds=[3000]
filter_approaches = ['probability']
df_names = ['abalone', 'aust_credit', 'balance_scale', 'bank', 'banknote', 
            'biodeg', 'breast cancer', 'car', 'credit', 'cryotherapy', 'ecoli', 
            'forest', 'german', 'glass', 'haberman', 'internet', 'iris', 'kohkiloyeh',
            'liver', 'magic', 'mamographic', 'nurse', 'occupancy', 'pima', 'seismic', 
            'spambase', 'tic-tac-toe', 'vegas', 'winery', 'zoo']
df_names = ['iris']
number_of_estimators=100
fixed_params={}
num_of_iterations=3

e = ExperimentSetting(branch_probability_thresholds,df_names,
                      number_of_estimators,fixed_params, num_of_iterations)

# 这个删除pickles_200trees文件夹下的所有文件是我自己找的代码，如果要
folder_path = 'pickles_200trees'
files = glob.glob(os.path.join(folder_path, '*')) # 收集文件夹中的所有文件

# 逐个删除文件 
for file in files:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")
tic = time.time()
e.run()
toc = time.time()
elapsed_time = toc - tic
print('Time elapsed: ', elapsed_time)