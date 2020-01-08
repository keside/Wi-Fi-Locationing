# Indoor localization through Wi-fi footprint 

# Summary
GPS is used for locating outdoor localization and cannot be used indoors but it's possible to location a person indoors using Wi fi footprint.

# Dataset
The Dataset can be easily download from the UCI MAchine Learning Repository in here: http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc. Visit the website for more descriptive information about the dataset . Training dataset: Features: 529, Observations: 19,937 and Validation Dataset: 1111 Obeservations and 529 Features

# PreProcessing
# NearZeroVariance 
# PCA
I used Principle component analysis for dimensional-reduction technique of the dataset to transform the number of possible correlated variables into a smaller number of uncorrelated variables.
# Normalization ie for scaling purpose

# Data Visualization

# Analysis
Longitude, Latitude, Relative position and Altitude are the only values I need in order to locate someone.The main purpose is to be able to locate a person in building in order to show him his position on a map and be able to give some instructions to move inside the building.

# Regression and Classification Machine Learning Modelling
Its used to predict numerical and catagorical variable in a feature of oberservations
Regression for Longitude and Latitude using Kknn, RF, SVM algorithm 
Classification for Floor and Building using RF, SVM, C5.0 




