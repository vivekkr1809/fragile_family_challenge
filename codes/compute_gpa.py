"""
This is the module file to train and predict on gpa data.

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
import matplotlib.pyplot as plt

# Import the modules for regression
import sklearn.linear_model as sklinear
from sklearn import feature_selection
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn  import ensemble
from sklearn import metrics
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation

def prediction_specific_preprocessing(background_data):
	"Modify the brackground data specific to this prediction"
	#num = background_data._get_numeric_data()
	#num[num < 0] = 1
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

def cross_validate_model(X_train, Y_train):
	"""
	Here we perform cross validation of models to choose the best one.
	"""
	# Divide the training and testing data
	train, test, y_actual, y_predict = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

	# List the regression methods to use.
	clf_random_forest = ensemble.RandomForestRegressor(n_estimators=50)
	clf_adaboost_reg = ensemble.AdaBoostRegressor(n_estimators=50)
	clf_lasso_larscv = sklinear.LassoLarsCV(cv=9)
	clf_ridge = sklinear.RidgeCV()
	clf_elastic_net = sklinear.ElasticNet()
	clf_extra_tree = ensemble.ExtraTreesRegressor(n_estimators=50)
	clf_mlpr = neural_network.MLPRegressor(solver='adam')

	# Add the above methods in an array
	# More ameable for looping
	methods = [clf_random_forest, clf_adaboost_reg, clf_lasso_larscv, clf_elastic_net, clf_extra_tree, clf_mlpr]
	methods_label = ['clf_random_forest', 'clf_adaboost_reg', 'clf_lasso_larscv', 'clf_elastic_net', 'clf_extra_tree', 'clf_mlpr']

	method_mse = np.zeros((len(methods),1))
	# Fit and predict for each method
	for i in range(len(methods)):
		methods[i].fit(train, y_actual)
		method_predict = methods[i].predict(test)
		method_mse[i] = metrics.mean_squared_error(y_predict, method_predict)
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
	gpa : 37.22

	Note : Code taken as is from homework 1 submission
	"""

	# feature selction-mutual info
	MIC=[]
	# Mutual info criteria
	MIC=feature_selection.mutual_info_regression(x_train, y_train)
	# get most descriptive features (here called good features)
	good_features=[]
	scores = []
	for k in range(len(MIC)):
		scores.append(MIC[k])
		if MIC[k] > 0.1: # Criteria for deciding that feature should be included
			good_features.append(k)
	# Adapt the training and testing matrices to good features
	x_train=x_train[:,good_features]
	x_test=x_test[:,good_features]
	print(len(good_features))
	return x_train, x_test, scores

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
	percent = 0.99
	pca = PCA(percent)
	# Fit the training data
	pca.fit(X_train, Y_train)
	print('Number of components required to explain %f of variance are %d' %(percent, pca.n_components_))

	# Apply mapping to both training and testing data
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	return X_train, X_test


def perform_bootstrapping(x_train, y_train):
	"We perform bootstrapping in this function"
	plt.figure(figsize=(12,8))
	x = np.linspace(1, len(y_train) ,num=len(y_train))
	n_loops = 100
	y_pred = []
	for i in range(0, n_loops):
		train, test, y_actual, y_predict = train_test_split(x_train, y_train, test_size=0.5)   
		rf = ensemble.RandomForestRegressor(n_estimators=50, max_depth=5)
		rf.fit(train, y_actual)
		y_pred.append(rf.predict(x_train))

	y_pred = np.transpose(y_pred)
	np.savetxt("bootstrapping.csv", y_pred, delimiter=",")
	for i in range(0, n_loops):
		plt.plot(x,y_pred[:,i],color='grey',linewidth = 2.)
	plt.scatter(x,y_train, marker='o', color='red', zorder=4)
	
	plt.savefig('bootstrapping.eps',bbox_inches='tight', format='eps', dpi=1000)
	plt.show()
	return 0


def prediction_step(background_train, background_test, gpa_data, challengeID_train):
	
	# Convert the background training and testing to numpy arrays
	background_train_np = background_train.as_matrix()
	background_train_np = np.asmatrix(background_train_np)

	background_test_np = background_test.as_matrix()
	background_test_np = np.asmatrix(background_test_np)

	gpa_data_np = gpa_data.as_matrix()
	gpa_data_np = np.asmatrix(gpa_data_np)
	gpa_data_np = np.ravel(gpa_data_np)

	# Perform fecture selection to reduce the number of
	# required features
	#background_train_np, background_test_np, scores = select_feature(background_train_np, background_test_np, gpa_data_np)

	#np.savetxt("feature_selection_gpa_scores.csv", scores, delimiter=",")

	# Perform principal component analysis
	#background_train_np, background_test_np = perform_pca(background_train_np, background_test_np, gpa_data_np)

	# Perform Cross Validation
	#position= cross_validate_model(background_train_np, gpa_data_np)

	perform_bootstrapping(background_train_np, gpa_data_np)

	####################################################
	## Set up the same methods used in cross validation
	## Fitting twice gives an error hence this way
	####################################################
	# List the regression methods to use.
	clf_random_forest = ensemble.RandomForestRegressor(n_estimators=50)
	clf_adaboost_reg = ensemble.AdaBoostRegressor(n_estimators=50)
	clf_lasso_larscv = sklinear.LassoLarsCV(cv=9)
	clf_ridge = sklinear.RidgeCV()
	clf_elastic_net = sklinear.ElasticNet()
	clf_extra_tree = ensemble.ExtraTreesRegressor(n_estimators=50)
	clf_mlpr = neural_network.MLPRegressor(solver='adam')

	# Add the above methods in an array
	# More ameable for looping
	methods = [clf_random_forest, clf_adaboost_reg, clf_lasso_larscv, clf_elastic_net, clf_extra_tree, clf_mlpr]
	methods_label = ['clf_random_forest', 'clf_adaboost_reg', 'clf_lasso_larscv', 'clf_elastic_net', 'clf_extra_tree', 'clf_mlpr']
	
	# Add the position of the classifier
	method = methods[position]
	method_label = methods_label[position]

	print('The chosen method is : %s' %(method_label))

	# Predict based on the chosen method
	method.fit(background_train_np, gpa_data_np)
	predict_gpa = method.predict(background_test_np)
	filename = 'predict_gpa_'+method_label+'.csv'

	# If the file exists, delete the file
	if os.path.isfile(filename) :
		os.remove(filename)
	# Write to the file
	for i in range(len(predict_gpa)):
		file = open(filename,"a+")
		file.write("%f \r\n" % (predict_gpa[i]))

	file.close()

def gpa_calculation(path, train_data, background_data, challengeID_train):

	print('We are computing GPA')
	start_time = time.time()

	"""
	Step : Perform some prediction specific data processing
	"""
	background_data = prediction_specific_preprocessing(background_data)


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
	Step : Extract and impute the gpa data.
	1. Extract gpa from training data
	2. Impute with mean, median and mode.
	3. Choose the best imputing method based on cross validation.
	"""
	#print(train_data)
	gpa_data = train_data[train_data.columns[1]].copy()
	gpa_data = gpa_data.fillna(gpa_data.mean())
		
	# For this problem we would have to predict everything.
	# Hence the test case is the complete data set
	
	background_test = background_data.copy()

	"""
	Step : Predict the gpa. 
	We have to predict the gpa of all the cases and not only the withheld cases
	from the training set.
	"""
	prediction_step(background_train, background_test, gpa_data, challengeID_train)

	print 'GPA Runtime:', str(time.time() - start_time)


if __name__ == '__main__':
	print('This is the module file for calculating the gpas.\n\
		You must have receievd the main file and a readme to run the entire project.\n\
		Please contact the Author(s) if this is the only file you have.')
