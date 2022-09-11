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
import matplotlib.pyplot as plt

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
import compute_gpa as compute_gpa
import compute_grit as compute_grit
import compute_hardship as compute_hardship
import compute_eviction as compute_eviction
import compute_layoff as compute_layoff
import compute_job_training as compute_job_training

def preprocessing_data(inputcsv, home_path):

	start_time = time.time()

	# read input csv - takes time
	background_data = pd.read_csv(inputcsv, index_col = False, low_memory=False)
	# Fix date bug
	background_data.cf4fint = ((pd.to_datetime(background_data.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)

	# Replace Empty values with NAN
	background_data = background_data.replace(r'^\s+$', np.nan, regex=True)

	# replace NA's with mode
	background_data = background_data.fillna(background_data.mode().iloc[0])
	# if still NA, replace with -10
	background_data = background_data.fillna(value=-10)
	# replace some important strings with numbers
	background_data = background_data.replace(to_replace='Other', value = -11)
	background_data = background_data.replace(to_replace='Missing', value = -12)
	background_data = background_data.replace(to_replace='never happened', value = -13)
	background_data = background_data.replace(to_replace='head start', value = -14)
	background_data = background_data.replace(to_replace='state funded', value = -15)
	background_data = background_data.replace(to_replace='city funded', value = -16)
	background_data = background_data.replace(to_replace='time out', value = -17)
	background_data = background_data.replace(to_replace='m', value = -18)
	background_data = background_data.replace(to_replace='own children', value = -19)
	background_data = background_data.replace(to_replace='city/state welfare child care funds', value = -20)
	background_data = background_data.replace(to_replace='family independence agency', value = -21)
	background_data = background_data.replace(to_replace='state welfare', value = -22)

	# Extract the columns which are string
	background_string = background_data.select_dtypes(include='object')

	# Extract the values of the column
	column_values = list(background_string.columns.values)

	# Convert the columns with just numbers to float
	for i in range(len(column_values)):
		try :
			background_data[column_values[i]] = background_data[column_values[i]].astype(float)
		except ValueError:
			pass

	# Read the background data, excluding the columns which are still object type
	background_data = background_data.select_dtypes(exclude='object')

	# write filled outputcsv
	background_data.to_csv(path+'background_data.csv', index=False)

	print 'Preprocessing Runtime:', str(time.time() - start_time)

	return background_data

def main():

	# Get the home path
	home_path = os.getenv("HOME")

	start_time = time.time()

	# Path to the folder where you have data
	# Should contain the following files;
	# 1. train.csv
	# 2. background.csv

	path = 'C:\Users\Victor Charpentier\Google Drive\PhD\cos424\Homework2\FFChallenge_v4/'

	# Read the train csv file into pandas dataframe
	train_data = pd.read_csv(path+'train.csv', index_col=False, low_memory=False)

	# Extract the challengeID from the training data
	# The rows corresponding to the challengeIDs will be extracted for creating the
	# background training data
	if False:
		challengeID_train = train_data[train_data.columns[0]].copy()
		challengeID_train.to_csv('challengeID_train.csv', index=False)
	else:
		challengeID_train = pd.read_csv('challengeID_train.csv', header=None, index_col=False, low_memory= False)

	if False:
		# Fill the missing data in the background data file
		# And perform some data cleaning
		# Since this is a slow and common process we are performing in the main file
		background_data = preprocessing_data(path+'background.csv', home_path)
	else:
		# This reads a panda-data frame
		background_data = pd.read_csv(path+'background_data.csv', index_col=False, low_memory=False)

		# This reads the already prepared numpy file
		#background_data = np.genfromtxt(path+'background_NoConstant_fillNeg.csv', delimiter = ',')
		#use_pandas = 0

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
	# compute_gpa.gpa_calculation(path, train_data, background_data, challengeID_train)

	# Call the module to compute and predict the Grit
	# compute_grit.grit_calculation(path, train_data, background_data, challengeID_train)

	# Call the module to train and predict material hardship
	# compute_hardship.hardship_calculation(path, train_data, background_data, challengeID_train)

	# Call the module to compute and predict the eviction
	compute_eviction.eviction_calculation(path, train_data, background_data, challengeID_train)

	# Call the module to compute and predict the eviction
	compute_layoff.layoff_calculation(path, train_data, background_data, challengeID_train)

	# Call the module to compute and predict the eviction
	compute_job_training.job_training_calculation(path, train_data, background_data, challengeID_train)

	print 'Total Runtime:', str(time.time() - start_time)

if __name__ == '__main__':
	main()
