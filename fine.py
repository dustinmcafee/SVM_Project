#!/usr/bin/python3
"""
Name: Dustin Mcafee
Fine grid search using OVA SVM and input data set, refer to README
"""

from sklearn import preprocessing               	# For standardizing the dataset
from sklearn import svm					# OVO
from sklearn.multiclass import OneVsRestClassifier      # OVA
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import math
import sys

def f1(y_true, y_pred): return f1_score(y_true, y_pred, average='macro')
def precision(y_true, y_pred): return average_precision_score(y_true, y_pred, average='macro')
def roc(y_true, y_pred): return roc_auc_score(y_true, y_pred, average='macro')


#    Evaluates a representation using a Linear SVM
#    It uses 6-fold cross validation for selecting the C parameter
#    :param train_data:
#    :param train_labels:
#    :param test_data:
#    :param test_labels:
#    :param n_jobs:
#    :return: the test accuracy

def evaluate_svm_OVA(train_data, train_labels, test_data, test_labels, parameters, n_jobs=-1):
	# Scale data to 0-1
	scaler = preprocessing.MinMaxScaler()
	train_data = scaler.fit_transform(train_data)	# Fit transform to training data
	test_data = scaler.transform(test_data)		# Transform Testing data based on training fit

	# Binarize labels
	lb = preprocessing.LabelBinarizer()
	train_labels = lb.fit_transform(train_labels)
	test_labels = lb.fit_transform(test_labels)

	scoring = {'prec': make_scorer(precision), 'roc' : make_scorer(roc)}
	model = OneVsRestClassifier(svm.SVC(max_iter = 1000000))
	clf = model_selection.GridSearchCV(model, parameters, n_jobs=n_jobs, cv=6, verbose=True, return_train_score=True, scoring=scoring, refit='prec')
	clf.fit(train_data, train_labels)
	i = 0
	for x in clf.cv_results_['params']:
		print(x, "mean train precision:", clf.cv_results_['mean_train_prec'][i], "+/-", (2*clf.cv_results_['std_train_prec'][i]), "| mean train AUC ROC:", clf.cv_results_['mean_train_roc'][i], "+/-", (2*clf.cv_results_['std_train_roc'][i]))
		i += 1

def main():
	trainData = np.genfromtxt('input/vowel/train/TrainingData.txt', delimiter=',', dtype=float)
	testData = np.genfromtxt('input/vowel/test/TestingData.txt', delimiter=',', dtype=float)

	trainCat = trainData[:,-1].copy()
	testCat = testData[:,-1].copy()

	trainData = np.delete(trainData, -1, 1)
	testData = np.delete(testData, -1, 1)

	kernel = ['rbf']
	if kernel[0] == 'rbf':
		parameters_ova = {'estimator__kernel': kernel, 'estimator__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'estimator__gamma' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
	elif kernel[0] == 'sigmoid' or kernel[0] == 'linear':
		parameters_ova = {'estimator__kernel': kernel, 'estimator__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'estimator__gamma' : ['auto']}
	else:
		parameters_ova = {'estimator__kernel': kernel, 'estimator__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'estimator__gamma' : ['auto'], 'estimator__degree' : [1, 2, 3, 4, 5]}

	evaluate_svm_OVA(trainData, trainCat, testData, testCat, parameters_ova)


main()
