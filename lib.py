import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import os





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
    if dropnas:
        df = df.loc[~df.artist_genre.isna()]
    df.artist_genre = df.artist_genre.str.replace(r'[', '')
    df.artist_genre = df.artist_genre.str.replace(r']', '')
    df.artist_genre = df.artist_genre.str.replace(r'\'', '')
    df.artist_genre = df.artist_genre.str.replace(r'\"', '')
    df.artist_genre = df.artist_genre.str.replace(r' ', '')
    
    return df

def dummify(data, categ_cols, dropfirstcond=True):
    for col in categ_cols:
        data[col].astype('category')
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=dropfirstcond)
        data = data.join(dummies)
        data.drop(columns=[col], inplace=True)
        
    return data
             
              
def classify_genres_ALL(input_df, col, drop_genre_col=True):
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




def get_many_genres(df, broad_genre_list, unique_genre_list):
    gdf = df.copy()
    
    uniques_to_drop = unique_genre_list
    
    main_genre_list=[]

    for genre_list in broad_genre_list:
        main_genre = genre_list[0]
        main_genre_list.append(main_genre)

    new_broad_genre_list=[]
    for genre_list in broad_genre_list:
        genre_stems = genre_list
        print('our genre_stem_list', genre_stems)
        new_genre_sub_list=[]
        for sub in genre_stems:
#             print('sub ', sub)
            for g in unique_genre_list: 
                if sub in g:
#                     print('str ', sub)
                    print('____________we append ', g)
                    new_genre_sub_list.append(g)
        print('FINISHED ONE!', new_genre_sub_list)
        new_broad_genre_list.append(new_genre_sub_list)
    
    
    for g in main_genre_list:
        uniques_to_drop.remove(g)
    
    for g in uniques_to_drop:
        if g not in gdf.columns:
            uniques_to_drop.remove(g)
    
                   
    return new_broad_genre_list, uniques_to_drop
                  
def classify_genres_NARROW(df, broad_genre_list, unigen, drop_uniques=True):
    new_df = df.copy()
    
    main_genre_list=[]
    
#     drop_list = unigen
    
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
    
#     print(drop_list)
   
    if drop_uniques:
        for genre in final_drop_list:
            new_df.drop(columns=genre[0], inplace=True)
    
    return new_df


def full_clean(data, dropnas=True, drop_unique_genres=True, col_name='artist_genre'):
    new_df = strip_genre_col(data, dropnas=dropnas)
    new_df, unique_list = classify_genres_ALL(input_df = new_df, col = col_name, 
                                              drop_genre_col=True)
    final_df = classify_genres_NARROW(new_df, broad_genre_list = broad_genres, 
                                      unique_genre_list = unique_list, 
                                      drop_uniques=drop_unique_genres)
    
    return final_df


              
# ______________ VISUALIZATION FUNCTIONS ______________________

def eda_visuals(data, scatter_matrix=False, 
                corr_matrix=False, 
                dists = False):
    pass


# _____________ PIPELINE FUNCTIONS __________________________

def model_pipe(model_list, cv_num=5):
#     model_results=pd.Dataframe()
    
#     pipe = Pipeline([('mms', scaler),
#                      ('pca', PCA(n_components=10)),
#                      ('tree', tree.DecisionTreeClassifier(random_state=123))])
    
#     # Create the grid parameter
#     grid = [{'svm__kernel': ['poly', 'sigmoid'],
#              'svm__C': [0.01, 1, 100],
#              'svm__degree0': [2,3,4,5],
#              'svm__gamma': [0.001, 0.01]}]

#     # Create the grid, with "pipe" as the estimator
#     gridsearch = GridSearchCV(estimator=pipe,
#                       param_grid=grid,
#                       scoring='accuracy',
#                       cv=cv_num)
    pass

def model_all(x, y, cv_num=5, show_tree=True, save_tree=False):
#     model_results = pd.DataFrame()
    
#     scaler = MinMaxScaler()
    
    
    
#     #running and 
#     tree_clf = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 10)
#     tree_clf.fit(x,y)
    
#     if show_tree:
#         export_graphviz(
#                     tree_clf,
#                     feature_names = x.columns,
#                     class_names = np.unique(y).astype('str'),
#                     rounded = True,
#                     filled = True)
        
#         if save_tree:
#             PROJECT_ROOT_DIR = "."
#             CHAPTER_ID = "decision_trees"
#             IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
#             os.makedirs(IMAGES_PATH, exist_ok=True)
#             def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#                 path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#                 print("Saving figure", fig_id)
#                 if tight_layout:
#                     plt.tight_layout()
#                 plt.savefig(path, format=fig_extension, dpi=resolution)
#             export_graphviz(
#                     tree_clf,
#                     out_file = os.path.join(IMAGES_PATH, "tracks.dot"),
#                     feature_names = X.columns,
#                     class_names = np.unique(y).astype('str'),
#                     rounded = True,
#                     filled = True)
            
#             Source.from_file(os.path.join(IMAGES_PATH, "tracks.dot"))
    
    
    pass
    
    
    