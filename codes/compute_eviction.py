"""
This is the module file to train and predict on eviction data.

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

def prediction_specific_preprocessing(background_data, use_pandas):
	"Modify the brackground data specific to this prediction"
	if use_pandas:
		num = background_data._get_numeric_data()
		num[num < 0] = 1

	return background_data

def extract_backgorund_train(background_train, background_data, challengeID_train, use_pandas, path):
	"""
	Split the background data into training and testing.
	In this case the trainging includes the challengeID for
	which we have data in the train.csv. The test/prediction is
	performed on the whole data
	"""
	# Copy the background data into background train if challengeID matches
	if use_pandas:
		background_train = background_data.iloc[challengeID_train]
	else:
		background_train = np.genfromtxt(path+'background_NoConstant_fillNeg_train.csv', delimiter = ',')
	return background_train

def cross_validate_model(X_train, Y_train):

	"""
	Here we perform cross validation of models to choose the best one.
	"""
	# Divide the training and testing data
	train, test, y_actual, y_predict = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
	# train_n, test_n, y_actual_n, y_predict_n = train_test_split(X_train, Y_train, test_size=0.4, random_state=42)

	# # Add one hot encoder
	# rf = ensemble.RandomForestClassifier(n_estimators=150, max_depth=5)
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
	clf_random_forest = ensemble.RandomForestClassifier(n_estimators=150)
	clf_adaboost = ensemble.AdaBoostClassifier(n_estimators = 150)
	clf_mlpc = neural_network.MLPClassifier()
	clf_extra_tree = ensemble.ExtraTreesClassifier(n_estimators=150, bootstrap=True)

	# Add the above methods in an array
	# More ameable for looping
	methods = [clf_quaddis, clf_random_forest, clf_adaboost, clf_mlpc, clf_extra_tree]
	methods_label = ['clf_quaddis', 'clf_random_forest', 'clf_adaboost', 'clf_mlpc', 'clf_extra_tree']

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
	eviction : 37.22

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
def select_k_best(X_train, X_test, Y_train):
	"""
	This function selects the best k features using chi2
	"""
	k_features = 5000
	# Check if the number of features asking for exist
	# If not then ask for all
	ch2 = feature_selection.SelectKBest(feature_selection.chi2,k= k_features)
	
	X_train = ch2.fit_transform(X_train, Y_train)
	X_test = ch2.transform(X_test)
	return X_train , X_test

def perform_one_hotencoding(X_train, X_test, Y_train):

	train, test, y_actual, y_predict = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

	rf = ensemble.RandomForestClassifier(n_estimators=150, max_depth=5)
	rf_enc = OneHotEncoder()
	rf_lm = sklinear.LogisticRegression()
	rf.fit(train, y_actual)
	rf_enc.fit(rf.apply(train))
	rf_lm.fit(rf_enc.transform(rf.apply(test)), y_predict)
	y_predict_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))

	return y_predict_rf_lm



def prediction_step(path, background_train, background_test, eviction_data, challengeID_train, use_pandas):
	
	# We apply transform to both the training and test set
	#background_train_np = enc.transform(background_train_np)
	#background_test_np = enc.transform(background_test_np)

	# Convert the background training and testing to numpy arrays

		
	if use_pandas:
		background_train_np = background_train.as_matrix()
		background_train_np = np.asmatrix(background_train_np)
		background_test_np = background_test.as_matrix()
		background_test_np = np.asmatrix(background_test_np)
	else:
		background_train_np = background_train
		background_test_np = background_test
	
	eviction_data_np = eviction_data.as_matrix()
	eviction_data_np = np.asmatrix(eviction_data_np)
	eviction_data_np = np.ravel(eviction_data_np)



	# Perform fecture selection to reduce the number of
	# required features
	#background_train_np, background_test_np = select_feature(background_train_np, background_test_np, eviction_data_np)

	# Select k-best features
	#n background_train_np, background_test_np = select_k_best(background_train_np, background_test_np, eviction_data_np)

	# Perform principal component analysis
	# background_train_np, background_test_np = perform_pca(background_train_np, background_test_np, eviction_data_np)

	# Perform principal random tree embedding
	#predict_eviction = perform_one_hotencoding(background_train_np, background_test_np, eviction_data_np)

	# Perform Cross Validation
	# Choose the method to perform the actual prediction using the best performing
	# scheme
	position = cross_validate_model(background_train_np, eviction_data_np)

	####################################################
	## Set up the same methods used in cross validation
	## Fitting twice gives an error hence this way
	####################################################
	# List the regression methods to use.
	clf_quaddis = discriminant_analysis.QuadraticDiscriminantAnalysis()
	clf_random_forest = ensemble.RandomForestClassifier(n_estimators=150)
	clf_adaboost = ensemble.AdaBoostClassifier(n_estimators = 150)
	clf_mlpc = neural_network.MLPClassifier()
	clf_extra_tree = ensemble.ExtraTreesClassifier(n_estimators=150, bootstrap=True)

	# Add the above methods in an array
	# More ameable for looping
	methods = [clf_quaddis, clf_random_forest, clf_adaboost, clf_mlpc, clf_extra_tree]
	methods_label = ['clf_quaddis', 'clf_random_forest', 'clf_adaboost', 'clf_mlpc', 'clf_extra_tree']

	method = methods[position]
	method_label = methods_label[position]

	print('The chosen method is : %s' %(method_label))

	# Predict based on the chosen method
	method.fit(background_train_np, eviction_data_np)
	predict_eviction = method.predict_proba(background_test_np)
	filename = 'predict_eviction_'+method_label+'.csv'
	if os.path.isfile(filename) :
		os.remove(filename)

	for i in range(len(predict_eviction)):
		file = open(filename,"a+")
		file.write("%f \r\n" % (predict_eviction[i,1]))

	file.close()


def eviction_calculation(path, train_data, background_data, challengeID_train, use_pandas):

	print('We are computing eviction')

	start_time = time.time()
	"""
	Step : Perform some prediction specific data processing
	"""
	background_data = prediction_specific_preprocessing(background_data, use_pandas)


	"""
	Step : Extract the rows from the huge matrix corresponding to the id in the
	        training data
	1. Extract the rows of the background data
	2. Save the background data on which we would train
	"""

	# Initialize an empty data frame
	if use_pandas:
		background_train = pd.DataFrame()
	else:
		background_train = []
	background_train = extract_backgorund_train(background_train, background_data, challengeID_train, use_pandas, path)

	# The background test data is the whole data set so we do not create another file

	"""
	Step : Extract and impute the eviction data.
	1. Extract eviction from training data
	2. Impute with mode as this is a classification problem.
		Mean and Mode would not works.
	"""
	if use_pandas:
		eviction_data = train_data[train_data.columns[4]].copy()
		eviction_data = eviction_data.fillna(eviction_data.mode().iloc[0])
	else:
		train_data = train_data.dropna(thresh=3)
		eviction_data = train_data[train_data.columns[4]].copy()
		eviction_data = eviction_data.fillna(eviction_data.mode().iloc[0])



	
	# For this problem we would have to predict everything.
	# Hence the test case is the complete data set	

	"""
	Step : Predict the eviction. 
	We have to predict the eviction of all the cases and not only the withheld cases
	from the training set.
	"""
	prediction_step(path, background_train, background_data, eviction_data, challengeID_train, use_pandas)

	print 'eviction Runtime:', str(time.time() - start_time)


if __name__ == '__main__':
	print('This is the module file for calculating the evictions.\n\
		You must have receievd the main file and a readme to run the entire project.\n\
		Please contact the Author(s) if this is the only file you have.')
