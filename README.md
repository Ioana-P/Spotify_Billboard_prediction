## Executive Summary:
## Spotify Billboard Hit Prediction
Fadi Sarraf (FS) and Ioana Preoteasa (IP)

Using the USA billboard 100 dataset from 1990 onwards and data collected from Spotify's API, we built a series of classification models to predict whether a track will make it on the billboard. Is it hot or not?

We started by using four different classifier models (logistic regression, K-nearest neighbours, decision trees and support vector classifiers) through Scikit-Learn's library to achieve our baseline results for ROC AUC. We then chose our most successful model (decision trees) and implemented ensemble methods (Random Forest and VoteStacking) in order to push the success of our model further. 

Our most successful model+ensemble method was  ______ with a ROC AUC score of __ .

Next steps:
We would like to perform further modelling on a subset of data. For songs that did make it onto the Billboard, we want to predict their peak position and how longthey stay on the Billboard. 

index.ipynb - our final, technical notebook; also contains our data dictionary in section 1.
test_data.csv - final data set used for generating models and predictions
library.py - python file with all of our functions for cleaning and modelling stored
Spotify_billboard_prediction_ppt.pdf - pdf of our final presentation to stakeholders


