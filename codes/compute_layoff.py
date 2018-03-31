"""
This is the module file to train and predict on layoff data.

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

# Import the modules for regression
import sklearn.linear_model as sklinear
from sklearn import feature_selection
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn  import ensemble
from sklearn import discriminant_analysis
from sklearn import metrics
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

def preprocessing_data(inputcsv):

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

	# Replace some negative numbers with some positive values
	num = background_data._get_numeric_data()
	num[num < -4] = 1

	# Read the background data, excluding the columns which are still object type
	background_data = background_data.select_dtypes(exclude='object')

	print 'Preprocessing Runtime:', str(time.time() - start_time)

	return background_data

def extract_backgorund_train(background_train, background_data, challengeID_train):
	"""
	Split the background data into training and testing.
	In this case the trainging includes the challengeID for
	which we have data in the train.csv. The test/prediction is
	performed on the whole data
	"""
	# Copy the background data into background train if challengeID matches
	background_train = background_data.iloc[challengeID_train]
	return background_train

def extract_impute_layoff(layoff_data, background_train):
	"""
	Impute the missing layoff.
	If a sofisticated method for imputation is desired.
	Currently not implemented.
	"""

	"""
	This is the first naive attempt at imputing the data. We impute
	the data using mean, median and mode. Choose the method based on MSE.
	"""

	"""
	Step : Impute the data using mean
	"""
	# replace NA's with mean, median and mode
	layoff_data_mean = layoff_data.fillna(layoff_data.mean()).copy()
	layoff_data_median = layoff_data.fillna(layoff_data.median()).copy()
	layoff_data_mode = layoff_data.fillna(layoff_data.mode().iloc[0]).copy()

	# As a starting point we remove all the object type columns from the
	# background information
	background_train = background_train.select_dtypes(exclude='object')	
	background_train_np = background_train.as_matrix()
	background_train_np = np.asmatrix(background_train)

	# Mean data
	layoff_data_mean_np = layoff_data_mean.as_matrix()
	layoff_data_mean_np = np.asmatrix(layoff_data_mean_np)
	layoff_data_mean_np = np.ravel(layoff_data_mean_np)
	mean_train, mean_test, mean_actual, mean_predict = train_test_split(background_train_np, layoff_data_mean_np, test_size=0.4, random_state=0)
	
	# Median data
	layoff_data_median_np = layoff_data_median.as_matrix()
	layoff_data_median_np = np.asmatrix(layoff_data_median_np)
	layoff_data_median_np = np.ravel(layoff_data_median_np)
	median_train, median_test, median_actual, median_predict = train_test_split(background_train_np, layoff_data_median_np, test_size=0.4, random_state=0)
	
	# Mode data
	layoff_data_mode_np = layoff_data_mode.as_matrix()
	layoff_data_mode_np = np.asmatrix(layoff_data_mode_np)
	layoff_data_mode_np = np.ravel(layoff_data_mode_np)
	mode_train, mode_test, mode_actual, mode_predict = train_test_split(background_train_np, layoff_data_mode_np, test_size=0.4, random_state=0)
	
	# Predict the training data using random classifier
	clf_random_forest = ensemble.RandomForestRegressor(n_estimators=100)

	# Fit with mean
	clf_random_forest.fit(mean_train, mean_actual)
	y_mean = clf_random_forest.predict(mean_test)
	mean_mse = metrics.mean_squared_error(mean_predict, y_mean)

	# Fit with median
	clf_random_forest.fit(median_train, median_actual)
	y_median = clf_random_forest.predict(median_test)
	median_mse = metrics.mean_squared_error(median_predict, y_median)
	
	# Fit with mode
	clf_random_forest.fit(mode_train, mode_actual)
	y_mode = clf_random_forest.predict(mode_test)
	mode_mse = metrics.mean_squared_error(mode_predict, y_mode)
	
	# Choose the method:
	min_mse = np.amin([[mean_mse, median_mse, mode_mse]])
	del layoff_data
	if mean_mse== min_mse:
		layoff_data = layoff_data_mean
		print('Mean was selected')
	elif median_mse==min_mse:
		layoff_data = layoff_data_median
		print('Median was selected')
	else:
		layoff_data = layoff_data_mode
		print('Mode was selected')

	layoff_data.to_csv('layoff/layoff_data.csv', index=False)

	return layoff_data

def cross_validate_model(X_train, Y_train):

	"""
	Here we perform cross validation of models to choose the best one.
	"""
	# Divide the training and testing data
	train, test, y_actual, y_predict = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)
	# train_n, test_n, y_actual_n, y_predict_n = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

	# Add one hot encoder
	# rf = ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)
	# rf_enc = OneHotEncoder()
	# rf_lm = sklinear.LogisticRegression()
	# rf.fit(train, y_actual)
	# rf_enc.fit(rf.apply(train))
	# rf_lm.fit(rf_enc.transform(rf.apply(train_n)), y_actual_n)
	# y_predict_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(test)))
	# mse_rf_lm = metrics.mean_squared_error(y_predict, y_predict_rf_lm[:,1])
	# print('MSE RandomForestClassifier followed by LogisticRegression is %f' %(mse_rf_lm))

	# List the regression methods to use.
	clf_quaddis = discriminant_analysis.QuadraticDiscriminantAnalysis()
	clf_logreg = sklinear.LogisticRegression(penalty='l1')
	clf_random_forest = ensemble.RandomForestClassifier(n_estimators=50)
	clf_adaboost = ensemble.AdaBoostClassifier(n_estimators = 50)
	clf_mlpc = neural_network.MLPClassifier()
	clf_extra_tree = ensemble.ExtraTreesClassifier(n_estimators=50, bootstrap=True)

	# Add the above methods in an array
	# More ameable for looping
	methods = [clf_quaddis, clf_logreg, clf_random_forest, clf_adaboost, clf_mlpc, clf_extra_tree]
	methods_label = ['clf_quaddis', 'clf_logreg', 'clf_random_forest', 'clf_adaboost', 'clf_mlpc', 'clf_extra_tree']

	method_mse = np.zeros((len(methods),1))
	# Fit and predict for each method
	for i in range(len(methods)):
		methods[i].fit(train, y_actual)
		method_predict = methods[i].predict_proba(test)
		method_mse[i] = metrics.mean_squared_error(y_predict, method_predict[:,1])
		print('MSE for %s while cross validation : %f' %(methods_label[i], method_mse[i]))

	# We return the method which has the minimum mse
	return np.argmin(method_mse)


def select_feature(x_train, x_test, y_train):
	"""
	This function reduces the number of features from the existing 
	g.t 10,000 to something manageable.
	Based on experience with feature selection in homework 1, we do
	not expect the selection to result in improved performance. But
	we expect a reduction in run-time.

	No feature Run Time 
	GPA : 320.58s
	Grit : 280.71
	Hardship : 288.05
	layoff : 37.22

	Note : Code taken as is from homework 1 submission
	"""
	# feature selction-mutual info
	MIC=[]
	# Mutual info criteria
	MIC=feature_selection.mutual_info_regression(x_train, y_train)
	# get most descriptive features (here called good features)
	good_features=[]
	for k in range(len(MIC)):
		if MIC[k] > 0.002: # Criteria for deciding that feature should be included
			good_features.append(k)
	# Adapt the training and testing matrices to good features
	x_train=x_train[:,good_features]
	x_test=x_test[:,good_features]
	print(len(good_features))
	return x_train, x_test

def perform_pca(X_train, X_test, Y_train):
	"""
	While performing Quadratic Discriminant Analysis collinear variables were
	identified. This function tries to remove as many such variables as possible
	"""
	# First we standardize the data set
	scaler = StandardScaler()
	# Fit on the training set
	scaler.fit(X_train)
	# We apply transform to both the training and test set
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	# Make principal component model
	# Set the amount of variance explained
	#percent = 0.99
	#pca = PCA(percent)
	# Fit the training data
	#pca.fit(X_train, Y_train)
	#print('Number of components required to explain %f of variance are %d' %(percent, pca.n_components_))

	# Apply mapping to both training and testing data
	#X_train = pca.transform(X_train)
	#X_test = pca.transform(X_test)

	return X_train, X_test

def perform_one_hotencoding(X_train, X_test, Y_train):

	train, test, y_actual, y_predict = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

	rf = ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)
	rf_enc = OneHotEncoder()
	rf_lm = sklinear.LogisticRegression()
	rf.fit(train, y_actual)
	rf_enc.fit(rf.apply(train))
	rf_lm.fit(rf_enc.transform(rf.apply(test)), y_predict)
	y_predict_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))

	return y_predict_rf_lm



def prediction_step(background_train, background_test, layoff_data, challengeID_train):
	
	# We apply transform to both the training and test set
	#background_train_np = enc.transform(background_train_np)
	#background_test_np = enc.transform(background_test_np)

	# Convert the background training and testing to numpy arrays
	background_train_np = background_train.as_matrix()
	background_train_np = np.asmatrix(background_train_np)

	background_test_np = background_test.as_matrix()
	background_test_np = np.asmatrix(background_test_np)

	layoff_data_np = layoff_data.as_matrix()
	layoff_data_np = np.asmatrix(layoff_data_np)
	layoff_data_np = np.ravel(layoff_data_np)



	# Perform fecture selection to reduce the number of
	# required features
	#background_train_np, background_test_np = select_feature(background_train_np, background_test_np, layoff_data_np)

	# Perform principal component analysis
	#background_train_np, background_test_np = perform_pca(background_train_np, background_test_np, layoff_data_np)

	# Perform principal random tree embedding
	#predict_layoff = perform_one_hotencoding(background_train_np, background_test_np, layoff_data_np)

	# Perform Cross Validation
	# Choose the method to perform the actual prediction using the best performing
	# scheme
	position = cross_validate_model(background_train_np, layoff_data_np)

	####################################################
	## Set up the same methods used in cross validation
	## Fitting twice gives an error hence this way
	####################################################
	# List the regression methods to use.
	clf_quaddis = discriminant_analysis.QuadraticDiscriminantAnalysis()
	clf_logreg = sklinear.LogisticRegression(penalty='l1')
	clf_random_forest = ensemble.RandomForestClassifier(n_estimators=50)
	clf_adaboost = ensemble.AdaBoostClassifier(n_estimators = 50)
	clf_mlpc = neural_network.MLPClassifier()
	clf_extra_tree = ensemble.ExtraTreesClassifier(n_estimators=50, bootstrap=True)

	# Add the above methods in an array
	# More ameable for looping
	methods = [clf_quaddis, clf_logreg, clf_random_forest, clf_adaboost, clf_mlpc, clf_extra_tree]
	methods_label = ['clf_quaddis', 'clf_logreg', 'clf_random_forest', 'clf_adaboost', 'clf_mlpc', 'clf_extra_tree']

	method = methods[position]
	method_label = methods_label[position]

	print('The chosen method is : %s' %(method_label))

	# Predict based on the chosen method
	method.fit(background_train_np, layoff_data_np)
	predict_layoff = method.predict_proba(background_test_np)
	filename = 'layoff/predict_layoff_'+method_label+'.csv'
	if os.path.isfile(filename) :
		os.remove(filename)

	for i in range(len(predict_layoff)):
		file = open(filename,"a+")
		file.write("%f \r\n" % (predict_layoff[i,1]))

	file.close()

def layoff_calculation(path, train_data, challengeID_train):

	print('We are computing layoff')

	start_time = time.time()
	
	# Fill the missing data in the background data file
	# And perform some data cleaning
	background_data = preprocessing_data(path+'background.csv')

	"""
	Step : Extract the rows from the huge matrix corresponding to the id in the
	        training data
	1. Extract the rows of the background data
	2. Save the background data on which we would train
	"""

	# Initialize an empty data frame
	background_train = pd.DataFrame()
	background_train = extract_backgorund_train(background_train, background_data, challengeID_train)

	# The background test data is the whole data set so we do not create another file

	"""
	Step : Extract and impute the layoff data.
	1. Extract layoff from training data
	2. Impute with mode as this is a classification problem.
		Mean and Mode would not works.
	"""
	layoff_data = train_data[train_data.columns[5]].copy()
	layoff_data = layoff_data.fillna(layoff_data.mode().iloc[0])
	
	
	# For this problem we would have to predict everything.
	# Hence the test case is the complete data set
	background_test = background_data.copy()


	"""
	Step : Predict the layoff. 
	We have to predict the layoff of all the cases and not only the withheld cases
	from the training set.
	"""
	#prediction_step(background_train, background_test, layoff_data, challengeID_train)

	print 'layoff Runtime:', str(time.time() - start_time)


if __name__ == '__main__':
	print('This is the module file for calculating the layoffs.\n\
		You must have receievd the main file and a readme to run the entire project.\n\
		Please contact the Author(s) if this is the only file you have.')

