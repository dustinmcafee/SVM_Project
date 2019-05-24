#!/usr/bin/python3
"""
Name: Dustin Mcafee
This statndardizes the dataset, refer to README
"""

from sklearn import preprocessing               # For standardizing the dataset
import csv
import numpy as np
import math
from numpy import genfromtxt
import sys




# Split data into Training and Testing sets
# Input: dataset: dataset to split
#        split: how many elements to give to the testing dataste
# Return training dataset, testing dataset
def dataSplit(dataset, split):
        #Copy the data as to not randomize the original set
        data = dataset.copy()
        np.random.shuffle(data)

        #Split the randomized dataset into a training dataset and a testing dataset
        valid, train = data[:split,:], data[split:,:]
        return train, valid

#Clean the data (Impute nan rows)
def imputeNAN(data, array):
	row_it = 0
	for row in data:
		col_it = 0
		num_imput = 0
		for elem in row:
			if(np.isnan(elem)):
				data[row_it, col_it] = array[col_it]
				num_imput = num_imput + 1
			col_it = col_it + 1
		if(num_imput > 0):
			print(num_imput, "Imputed in row", row_it)
		row_it = row_it + 1
	return data

def main():
	my_data = genfromtxt('input/vowel/vowel-context.data', delimiter=None, dtype=str)

	# Delete 1st 3 Columns: They are irrelevant
	my_data = np.delete(my_data, 0, axis=1)
	my_data = np.delete(my_data, 0, axis=1)
	my_data = np.delete(my_data, 0, axis=1)

	#Variables
	N = np.size(my_data, 0)
	dimensions = np.size(my_data, 1)
	print(dimensions, "Dimensions")
	print(N, "Observations")

	#Convert binary g/b categorical attribute to numerical 1/0
	#conv = {'g' : 1, 'b' : 0}
	#my_data[:,-1] = np.array([conv[x] for x in my_data[:,-1]])
	my_data = my_data.astype(float)
	my_data[:,-1] = my_data[:,-1].astype(int)

	#impute missing data
	mean = np.nanmean(my_data, axis=0)
	my_data = imputeNAN(my_data, mean)

	trainData, testData = dataSplit(my_data, 198)		# Testing Dataset is 20% of entire dataset

	np.savetxt("input/vowel/train/TrainingData.txt", np.matrix(trainData), delimiter=',', fmt='%3.4f')
	np.savetxt("input/vowel/test/TestingData.txt", np.matrix(testData), delimiter=',', fmt='%3.4f')

	return

main()
