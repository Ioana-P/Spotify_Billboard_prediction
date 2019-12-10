import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def clean(data, val_to_drop, prefix='PAY_', col_ind_list = [0,2,3,4,5,6]):
    for i in col_ind_list:
        col = prefix+str(i)
        data = data.loc[data[col]!=val_to_drop]
        
    return data

def dummify(data, categ_cols, dropfirstcond=True):
    for col in categ_cols:
        data[col].astype('category')
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=dropfirstcond)
        data = data.join(dummies)
        data.drop(columns=[col], inplace=True)
        
    return data

def eda_visuals(data, scatter_matrix=False, 
                corr_matrix=False, 
                dists = False):
    pass



    
