# Regression on the tabular data
This project contains code for building a model that predicts a target based on the proposed features.

# Prerequisites
The code requires the following libraries to be installed:

Pandas
Numpy
Scikit-learn

All the required libraries are in the requirements.txt file.

# Dataset
I used the given dataset files, the internship_train.csv for training and testing and internship_hidden_test.csv for predicting values.

# Model parameters
The model uses a Random Forest Regressor from scikit-learn with the following parameters:

max_depth: 20
n_estimators: 20
n_jobs: 1
They can be modified in the regression-tabular-data.py script to experiment with different configurations.
