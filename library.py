import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.externals.six import StringIO
from IPython.display import Image  
from graphviz import Source
import pydotplus


# List of main genres and their corresponding subgenres:
rock=['rock','punk','jamband','grunge']
hip_hop=["hiphop",'crunk','newjack']
rap=['rap','gfunk','drill','grime']
blues=['blues']
jazz=['jazz','ragtime','bossanova','bebop','swing','dixieland','contrabass']
r_n_b=['r&b','soul',"motown",'gospel','funk']
metal=['metal','hardcore','core','thrash']
country=['country','nashville','bluegrass','western','stompandholler','honkytonk']
pop=['pop','boyband','girlgroup','surf']
indie=['indie','neomellow','shoegaze',"singer-songwriter"]
classical=['classical','orchestral','stringquartet','ballet','ensemble','ballroom','choir','romanticera','military','band','baroque','viola','cello',
'orchestra','renaissance','opera']
oldies=['oldies','adultstandard']
latin=['latin','chunchaca','grupera','banda','dominican','champeta','cumbia','mexic','cuba','salsa','venezolano','brazil','marimba','rumba','tango']
arabic=['arabic','arab','ghazal','nasheed','dabke']
indian=['indian','bollywood','sitar']
eastern_european=['easterneuropean',]
carribean=['carribean','mambo','calypso']
african=['african','afrobeat','afro']
edm=['edm','downtempo','house','techno','electro','garage','jungle','trance','tronica','drumandbass','dnb','deepliquidbass','drillandbass',
'bassline','bigroom','skyroom','deepflow','bass']
japanese=['japanese','j-']
korean=['korean','k-']
idm=['idm','ambient','chill','trip']
folk=['folk','americana']
soundtrack=['soundtrack','movie','anime','videogame','broadway','westend','disney','theme']
children=['children','kinder','enfants','nursery','infantil','lullaby']
religious=['religious','chant','praise','ccm','worship','christian','spiritual','islam','anglican','cristiano','gaze']
celtic=['celtic','irish']
white_noise=['whitenoise','atmosphere']
alternative=['alternative','downshift','emo']
disco=['disco']
reggae=['reggae','dub','ska','soca','ragga','dancehall']
meditation=['meditation','chakra']
comedy=['comedy']
avant_garde=['lo-fi']
world=['world']

broad_genres = [rock,hip_hop,rap,blues,jazz,r_n_b,metal,country,pop,indie,classical,oldies,latin,arabic,indian,eastern_european,carribean, african,edm,japanese,korean,idm,folk,soundtrack,children,religious,celtic,white_noise,alternative,disco,reggae,meditation,comedy,avant_garde,world]

# ______________ CLEANING FUNCTIONS ______________________

def strip_genre_col(df, dropnas=True):
    # function takes in Spotify dataframe and expects a column called "artist_genre". It returns a dataframe with the genres rows stripped of square brackets, commas, quotes and spaces. If dropnas it also removes any values where there is no genre; returns dataframe
    if dropnas:
        df = df.loc[~df.artist_genre.isna()]
    df.artist_genre = df.artist_genre.str.replace(r'[', '')
    df.artist_genre = df.artist_genre.str.replace(r']', '')
    df.artist_genre = df.artist_genre.str.replace(r'\'', '')
    df.artist_genre = df.artist_genre.str.replace(r'\"', '')
    df.artist_genre = df.artist_genre.str.replace(r' ', '')
    
    return df

def dummify(data, categ_cols, dropfirstcond=True):
    # takes in dataframe and a list of categorical columns: creates binary dummy columns for each of those columns and drops original column; if dropfirstcond (recommended) then it also drops the first binary column; returns dataframe with new columns
    for col in categ_cols:
        data[col].astype('category')
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=dropfirstcond)
        data = data.join(dummies)
        data.drop(columns=[col], inplace=True)
        
    return data
             
              
def classify_genres_ALL(input_df, col, drop_genre_col=True):
    # Function for tranforming all the genre strings in one row into binary dummy columns.
    # takes in dataframe, a column with strings ('artist_genre'): splits the strings based on "," and creates a new column for each one. Returns a new dataframe and a list of all the unique terms that were found. If drop_genre_col it drops the original column at the end
    df = input_df.copy()
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
                
            else: 
                new_term+=j
              
    if drop_genre_col:
              df.drop(columns=[col], inplace=True)
                  
    return df, uniques

                  
def classify_genres_NARROW(df, broad_genre_list, unigen, drop_uniques=True):
    # Function that groups all the different unique genres into larger categories, based on the larger lists stored at the top of this .py file. Takes in a 
    # df - dataframe, 
    # broad_genre_list - the list of lists of genre_stems, 
    # unigen - the list of all genres in the original dataframe and 
    # drop_uniques - (bool) whether or not to remove the unique genres afterwards
    # Returns a dataframe
    new_df = df.copy()
    
    main_genre_list=[]
        
    new_broad_genre_list = []

    for genre_list in broad_genre_list:
        new_genre_list=[]
        for genre_stem in genre_list:
            for genre in unigen:
                if genre_stem in genre:
                    new_genre_list.append(genre)

        new_broad_genre_list.append(new_genre_list)
    
    final_drop_list=[]
    
    for genre_list in new_broad_genre_list:
        if len(genre_list)>1:
            main_genre = genre_list[0]
            final_drop_list.append(genre_list[1:])
        else:
            main_genre = genre_list
        main_genre_list.append(main_genre)
        new_df[str(main_genre)] = 0
        genre_col_list = genre_list   #[x for x in genre_list]
        for subgenre in genre_col_list:
            if subgenre in new_df.columns:
                new_df[str(main_genre)] += new_df[subgenre]
            
            
        new_df.loc[new_df[str(main_genre)]>1, str(main_genre)] = 1
    
    for genre in main_genre_list:
        if genre in final_drop_list:
            drop_list.remove(genre)
    
    if drop_uniques:
        for genre in final_drop_list:
            new_df.drop(columns=genre[0], inplace=True)
    
    return new_df

# _____________ TREE VISUAL FUNCTION __________________________

def plot_tree(input_df_columns, figsize=(20,20), tree_model,  max_depth=5)
    plt.figure(figsize=figsize)
    dot_data = StringIO()
    export_graphviz(model=tree_model, out_file=dot_data,  
                    filled=True, rounded=True, max_depth=max_depth,
                    special_characters=True, feature_names=input_df_columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    
    return 

              

# _____________ PIPELINE FUNCTIONS __________________________
def preproccessing():
    # purpose-built preprocessing function that calls Pipeline functions for processing numerical and categorical data
    # Returns a preprocessor object that is called inside the Pipeline object
    numeric_features = ['duration_ms', 'tempo', 'loudness', 'popularity']
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_features = ['key', 'mode', 'time_signature', 'artist_name']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))], verbose=True)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])
    return preprocessor


# _____________insert new functions below here!!!!

def run_all_models(x, y, model_list=['logistic', 'DecisionTree','KNN', 'poly-SVM'], 
                   scaler='MinMaxScaler', cv_num=5, test_size=.25):
    
    train_model_df = pd.DataFrame(index=model_list, columns = ['Accuracy_on_train', 'ROC_AUC_on_train'])
    test_model_df = pd.DataFrame(index=model_list, columns = ['Accuracy_on_test', 'ROC_AUC_on_test'])
    
    for model in model_list:
        model,scaler, train_score, train_rocauc, test_score, test_rocauc = model_pipe(x,y, model, scaler=scaler, cv_num=cv_num, test_size=test_size)
        train_model_df['Accuracy_on_train'][model] = train_score
        train_model_df['ROC_AUC_on_train'][model] = train_rocauc
        
        test_model_df['Accuracy_on_test'][model] = test_score
        test_model_df['ROC_AUC_on_test'][model] = test_rocauc
        
    return train_model_df, test_model_df
    

                   
def model_maker(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    numeric_features = ['duration_ms', 'tempo', 'loudness', 'popularity']
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_features = ['key', 'mode', 'time_signature', 'artist_name']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', KNeighborsClassifier(n_neighbors=3))])
                      
    
    clf.fit(X_train, y_train)
    clf.predict_proba(X_train)
    print('X_train Score: ', clf.score(X_train, y_train))
    print('(-----------------------)')
    clf.predict(X_test)
    print('X_test Score:  ', clf.score(X_test, y_test))
    
    return 



def model_pipe(x, y, model, scaler='MinMaxScaler', cv_num=5, test_size=.25):
   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    if scaler=='MinMaxScaler':
        numeric_features = ['duration_ms', 'tempo', 'loudness', 'popularity']
        numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())], verbose=True)
        
    if scaler=='StandardScaler':
        numeric_features = ['duration_ms', 'tempo', 'loudness', 'popularity']
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])
        
    
    if model=='logistic':
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(C=1, max_iter=1000,solver='lbfgs'))])

    
    
    if model=='poly-SVM':
        clf =Pipeline([
                    ('preprocessor', preprocessor),
                    ('svm_clf', SVC(kernel="poly", degree=3, gamma='auto', coef0=1, C=1))
                ])

        
    if model=='DecisionTree':
    
        preprocessor = ColumnTransformer(transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ])
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 100))])
        
        
    if model=='KNN':
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', KNeighborsClassifier(n_neighbors=3))])
    
    
    print(x_train.shape)
    print(x_test.shape)
    clf.fit(x_train,y_train)
    
    y_train_predict = clf.predict(x_train)
    
    train_score = clf.score(x_train,y_train)
    if model=='logistic':
        train_proba = clf.predict_proba(x_train)[:,0]
    else:
        train_proba = clf.predict(x_train)
    train_pred = clf.predict(x_train)
    train_rocauc = roc_auc_score(y_train, train_pred)
    
    test_score = clf.score(x_test, y_test)
    if model=='logistic':
        test_proba = clf.predict_proba(x_test)[:,0]
    else:
        test_proba = clf.predict(x_test)
    test_pred = clf.predict(x_test)
    test_rocauc = roc_auc_score(y_test, test_pred)
    
    
    fpr, tpr, thresh = roc_curve(y_test, test_pred)
    
    return model,scaler, train_score, train_rocauc, test_score, test_rocauc, fpr, tpr
