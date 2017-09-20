# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
    ## Inputs ##
    # yTrain - 1D numpy ndarray of length n

    ## Outputs ##
    # p - float
    p = 0
    count1 = yTrain[np.where(yTrain > 0)].size
    count0 = yTrain[np.where(yTrain == 0)].size
    ret =  count0*1.0/ (count0 + count1)*1.0
    return ret


# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
    ## Inputs ##
    # XTrain - (n by V) numpy ndarray
    # yTrain - 1D numpy ndarray of length n
    # alpha - float
    # beta - float

    ## Outputs ##
    # D - (2 by V) numpy ndarray

    D = np.zeros([2, XTrain.shape[1]])
    count1 = 1.0*yTrain[np.where(yTrain == 1)].size
    count0 = 1.0*yTrain[np.where(yTrain == 0)].size

    # complement the yTrain to get P(Xi|Y = 0)
    yTrain_0 = 1 - yTrain

    D = np.zeros([2, XTrain.shape[1]])
    x1y0 = XTrain.transpose().dot(yTrain_0)
    x1y1 = XTrain.transpose().dot(yTrain)

    const0 =  (1 / (count0 + beta_0 + beta_1 - 2))
    const1 =  (1 / (count1 + beta_0 + beta_1 - 2))

    D[0] = const0 * (x1y0 + beta_0*1.0 - 1)
    D[1] = const1 * (x1y1 + beta_0*1.0 - 1)

    return D


# The logProd function takes a vector of numbers in logspace
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    ## Inputs ##
    # x - 1D numpy ndarray

    ## Outputs ##
    # log_product - float

    log_product = sum(x)
    return log_product


# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
    ## Inputs ##
    # D - (2 by V) numpy ndarray
    # p - float
    # XTest - (m by V) numpy ndarray

    ## Outputs ##
    # yHat - 1D numpy ndarray of length m

    m = XTest.shape[0]
    yHat = np.ones(XTest.shape[0])

    for i in range(0,m):

        temp_store1 = XTest[i].dot(np.log(D[1])) + (1-XTest[i]).dot(np.log(1-D[1]))
        temp_store0 = XTest[i].dot(np.log(D[0])) + (1-XTest[i]).dot(np.log(1-D[0]))

        # temp_store1 = XTest[i].dot(D[1]) + (1 - XTest[i]).dot(1 - D[1])
        # temp_store0 = XTest[i].dot(D[0]) + (1 - XTest[i]).dot(1 - D[0])

        score1 = logProd([temp_store1, np.log(1-p)])
        score0 = logProd([temp_store0, np.log(p)])

        if score0 > score1:
            yHat[i] = 0.0
        else:
            yHat[i] = 1.0

    return yHat


# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
    ## Inputs ##
    # yHat - 1D numpy ndarray of length m
    # yTruth - 1D numpy ndarray of length m

    ## Outputs ##
    # error - float

    error = 0
    trues = 1.0 * yHat[np.where(yHat != yTruth)].size
    error = trues/yTruth.size
    return error
