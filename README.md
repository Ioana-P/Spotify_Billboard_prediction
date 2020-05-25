# Executive Summary:
# Spotify Billboard Hit Prediction
Fadi Sarraf (FS) and Ioana Preoteasa (IP)

Using the USA billboard 100 dataset from 1990 onwards and data collected from Spotify's API, we built a series of classification models to predict whether a track will make it on the billboard. Is it hot or not? Can we predict the commercial success of a track based on its 

We started by using four different classifier models (logistic regression, K-nearest neighbours, decision trees and support vector classifiers) through Scikit-Learn's library to achieve our baseline results for ROC AUC. We then chose our most successful model (decision trees) and implemented ensemble methods (Random Forest and VoteStacking) in order to push the success of our model further. 

Our most successful model method was a Decision Tree with a ROC AUC score of 0.8270044640013661 (on test data), shown below:

![Best model - Decision Tree](https://github.com/Ioana-P/Spotify_Billboard_prediction/fig/best_model_fig.jpg)



Future steps:
We would like to perform further modelling on a subset of data. For songs that did make it onto the Billboard, we want to predict their peak position and how longthey stay on the Billboard. 

* index.ipynb - our **final, technical notebook**; also contains our **data dictionary** in section 1.
* library_final.py - python file with all of our functions for cleaning and modelling stored
* eda1.ipynb - notebook containing our first exploration and cleaning of our track data

* data - directory containing our data files
    * class_tracks.csv - final data set used for generating models and predictions
    * data.csv - original,raw data file post merge of Billboard and Spotify data
    * Hot Stuff.csv - data on the Billboard 100 and whether tracks featured there or not, and for how long
    * Hot 100 Audio Features.xlsx - track metadata for the positive class, the tracks that did go to the Billboard 100
    * vis_data.csv - subset of numeric values in our dataset; used purely for some exploratory data visualizations
    * test_data.csv - test subset data file
    * tdf_final_test.csv - additional testing file

* fig - extra visualisation directory
* archive - admin documents from FIS and some extra notebooks
    * indexOLD.ipynb - first version of final notebook
    * dud.ipynb - 
    * module5_project_rubric.pdf - FIS Mod 5 rubric against which our project was scored
    * project_details - FIS Mod 5 project brief
    * lib.py & library.py previous iterations of our functions library file. **NOT** in use in main notebook
    
