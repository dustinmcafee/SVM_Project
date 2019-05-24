#!/usr/bin/python3
"""
Name: Dustin Mcafee
Coarse grid search using OVA SVM and input data set, refer to README
"""

from sklearn import preprocessing               	# For standardizing the dataset
from sklearn import svm					# OVO
from sklearn.multiclass import OneVsRestClassifier	# OVA
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import math
import sys

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
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

def evaluate_svm_OVO(train_data, train_labels, test_data, test_labels, parameters, n_jobs=-1):
	# Scale data to 0-1
	scaler = preprocessing.MinMaxScaler()
	train_data = scaler.fit_transform(train_data)	# Fit transform to training data
	test_data = scaler.transform(test_data)		# Transform Testing data based on training fit

	# Binarize labels
	lb = preprocessing.LabelBinarizer()
	train_labels = lb.fit_transform(train_labels)
	test_labels = lb.fit_transform(test_labels)

	scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn), 'accuracy': make_scorer(accuracy_score), 'prec': 'precision'}
	model = svm.SVC(max_iter=100000)		# OVO
	clf = model_selection.GridSearchCV(model, parameters, n_jobs=n_jobs, cv=3, verbose=True, return_train_score=True, scoring=scoring, refit='accuracy')
	clf.fit(train_data, train_labels)

	i = 0
	for x in clf.cv_results_['params']:
		print(x, "mean train accuracy:", clf.cv_results_['mean_train_accuracy'][i], "+/-", (2*clf.cv_results_['std_train_accuracy'][i]), "| mean train precision:", clf.cv_results_['mean_train_prec'][i], "+/-", (2*clf.cv_results_['std_train_prec'][i]))
		i += 1
	print(clf.best_score_)
	print(clf.best_params_)

def main():
	trainData = np.genfromtxt('input/vowel/train/TrainingData.txt', delimiter=',', dtype=float)
	testData = np.genfromtxt('input/vowel/test/TestingData.txt', delimiter=',', dtype=float)

	trainCat = trainData[:,-1].copy()
	testCat = testData[:,-1].copy()

	trainData = np.delete(trainData, -1, 1)
	testData = np.delete(testData, -1, 1)

	kernel = [str(sys.argv[1])]		# Can by 'poly', 'linear', 'sigmoid', and 'rbf'
	if kernel[0] == 'rbf':
		parameters_ovo = {'kernel': kernel, 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
		parameters_ova = {'estimator__kernel': kernel, 'estimator__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'estimator__gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
	elif kernel[0] == 'sigmoid' or kernel[0] == 'linear':
		parameters_ovo = {'kernel': kernel, 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma' : ['auto']}
		parameters_ova = {'estimator__kernel': kernel, 'estimator__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'estimator__gamma' : ['auto']}
	else:
		parameters_ovo = {'kernel': kernel, 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma' : ['auto'], 'degree' : [1, 2, 3, 4, 5]}
		parameters_ova = {'estimator__kernel': kernel, 'estimator__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'estimator__gamma' : ['auto'], 'estimator__degree' : [1, 2, 3, 4, 5]}

	#evaluate_svm_OVO(trainData, trainCat, testData, testCat, parameters_ovo)
	evaluate_svm_OVA(trainData, trainCat, testData, testCat, parameters_ova)


main()
