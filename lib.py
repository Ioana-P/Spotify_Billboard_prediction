import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ______________ CLEANING FUNCTIONS ______________________

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
             
              
def classify_genres_ALL(input_df, col, drop_genre_col=True):
    df = input_df
    uniques=[]
    for i, row in enumerate(df[col]):
        new_term=''
        for j in row:            
            if j==',':
                if new_term in uniques:
                    df.loc[df.index==i, new_term] = 1
                    new_term=''
                else:
                    uniques.append(new_term)
                    df[new_term] = 0
                    df.loc[df.index==i, new_term] = 1
                    new_term=''
                continue
            else: 
                new_term+=j
              
    if drop_genre_col:
              df.drop(columns=[col], inplace=True)
    
                
    return df, uniques   
              
def classify_genres_NARROW(df, broad_genre_list, unique_genre_list):
    pass
              
              
# ______________ VISUALIZATION FUNCTIONS ______________________

def eda_visuals(data, scatter_matrix=False, 
                corr_matrix=False, 
                dists = False):
    pass


              


    
