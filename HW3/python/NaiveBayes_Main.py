import os
import csv
import numpy as np
import NB

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'r') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

#Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# XTrain = np.array([1,1,0,1,0,0,0,1,1])
# XTrain.shape = (3,3)
# yTrain = np.array([0,1,1])
# XTest  = np.array([0,1,0,1,0,0,])
# XTest.shape = (2,3)
# yTest  = np.array([1,0])


# TODO: Test logProd function, defined in NB.py

# TODO: Test NB_XGivenY function, defined in NB.py
D = NB.NB_XGivenY(XTrainSmall,yTrainSmall,5,7)

# TODO: Test NB_YPrior function, defined in NB.py
p = NB.NB_YPrior(yTrainSmall)
#TODO: Test NB_Classify function, defined in NB.py
yHat = NB.NB_Classify(D,p,XTest)
# TODO: Test classificationError function, defined in NB.py
error = NB.classificationError(yHat,yTest)
print("Classification Error: {0}".format(error))
# TODO: Run experiments outlined in HW2 PDF
