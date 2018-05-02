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

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV


def memusage():
    print("memusage: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

filelist=os.listdir('data_str')

X_ = []
Y_ = []

memusage()

for f in filelist:
    im = numpy.array(imageio.imread('data_str/' + f))
    shape = im.shape
    im = im.reshape((shape[0]*shape[1]*shape[2]))
    X_.append(im)
    obj_type = f
    Y_.append(numpy.array(obj_type))
    print("Image loaded: " + f)
    memusage()



X = numpy.array(X_)
Y = numpy.array(Y_)

print("Numpy arrays created")
memusage()


#pca = PCA(n_components=10, svd_solver='randomized')
#X = pca.fit_transform(X)

print("PCA transformation applied")
memusage()
#m = GaussianNB()
m = LogisticRegression()

m.fit(X, Y)

im = imageio.imread(sys.argv[1])
V_ = numpy.array(im)
V = V_.reshape(1,-1)
#V = pca.transform(V)
print(m.predict(V))

memusage()
