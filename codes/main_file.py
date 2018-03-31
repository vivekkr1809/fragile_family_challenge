"""
This is the main source file for the fragile family challenge data.

Author(s) : Vivek Kumar
            vivekk@princeton.edu

            Victor Charpentier
            vc6@princeton.edu

Parts of code may have been provided by COS 424 staff. Such code portions are properly
credited.

Last Updated : 03-27-2018
"""

# Import the relevant packages/modules.
import numpy as np
import pandas as pd
import sympy as sp
import sys
import time
import os
import platform

# Import modules relevant to the regression/classification
import sklearn.linear_model as sklinear
from sklearn import metrics
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn  import ensemble
from sklearn import discriminant_analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

# Import the relevant files for predictions
# from gpa import compute_gpa as compute_gpa
# from grit import compute_grit as compute_grit
# from hardship import compute_hardship as compute_hardship
# from eviction import compute_eviction as compute_eviction
import compute_layoff as compute_layoff
# from job_training import compute_job_training as compute_job_training

def main():

	# Get the home path
	home_path = os.getenv("HOME")

	start_time = time.time()

	# Path to the folder where you have data
	# Should contain the following files;
	# 1. train.csv
	# 2. background.csv

	path = home_path+'/Dropbox/Princeton/2017-18/cos_424/homework/homework_2/FFChallenge_v4/'

	# Read the train csv file into pandas dataframe
	train_data = pd.read_csv(path+'train.csv', index_col=False, low_memory=False)
	
	# Extract the challengeID from the training data
	# The rows corresponding to the challengeIDs will be extracted for creating the
	# background training data
	if False:
		challengeID_train = train_data[train_data.columns[0]].copy()
		challengeID_train.to_csv('challengeID_train.csv', index=False)
	else:
		challengeID_train = pd.read_csv('challengeID_train.csv', index_col=False, low_memory= False)
	
	# Convert the data frame into numpy matrix
	challengeID_train = challengeID_train.as_matrix()
	challengeID_train = np.asmatrix(challengeID_train)
	# Convert the matrix into a 1-D array
	challengeID_train = np.ravel(challengeID_train)
	# Get the location of the data
	challengeID_train = challengeID_train-1
	# Convert the values to integer type
	challengeID_train = challengeID_train.astype(int)

	# Call the module to compute and predict the GPA

	# Pass the background data and call the function
	# compute_gpa.gpa_calculation(train_data, challengeID_train)

	# Call the module to compute and predict the Grit
	# compute_grit.grit_calculation(train_data, challengeID_train)

	# Call the module to train and predict material hardship
	# compute_hardship.hardship_calculation(train_data, challengeID_train)

	# Call the module to compute and predict the eviction
	# compute_eviction.eviction_calculation(train_data, challengeID_train)	

	# Call the module to compute and predict the eviction
	compute_layoff.layoff_calculation(path, train_data, challengeID_train)

	# Call the module to compute and predict the eviction
	# compute_job_training.job_training_calculation(train_data, challengeID_train)

	# Copy the prediction file
	prediction_file = pd.read_csv(path+'prediction.csv', index_col=False, low_memory=False)
	# Empty the prediction file
	prediction_file = prediction_file.iloc[0:0]

	print 'Total Runtime:', str(time.time() - start_time)

if __name__ == '__main__':
	main()

