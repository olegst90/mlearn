#! /usr/bin/python

import imageio
import os
import psutil
import numpy
import resource
import re
import sys
import time
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC



def memusage():
    print("memusage: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

X = numpy.load("X.opt.npy")
Y = numpy.load("Y.opt.npy")
V = numpy.load("V.opt.npy")

print("Numpy arrays loaded")
memusage()

#m = GaussianNB()
m = LogisticRegression()
m.fit(X, Y)
print(m.predict(V))
memusage()
