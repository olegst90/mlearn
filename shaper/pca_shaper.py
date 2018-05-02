#! /usr/bin/python

import imageio
import os
import psutil
import numpy
import resource
import re
import sys
import time
#import matplotlib.pyplot as plt
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

from sklearn.pipeline import Pipeline

filelist=os.listdir('data')

X_ = []
Y_ = []

for f in filelist:
    im = imageio.imread('data/' + f)
    #im_arr = numpy.array(im)
    #shape = im_arr.shape
    #im_arr = im_arr.reshape((shape[0],shape[1]*shape[2]))
    #pca.fit(im_arr)
    #X_.append(pca.transform(im_arr))
    X_.append(im)
    obj_type = re.split(r'\.',f)[0]
    Y_.append(numpy.array(obj_type))

X_ = numpy.array(X_)
Y_ = numpy.array(Y_)

shape = X_.shape
X = X_ .reshape((shape[0],shape[1]*shape[2]*shape[3]))
Y = Y_

print("Shape :")
print(X.shape)

pca = PCA(n_components=150, svd_solver='randomized')

X = pca.fit_transform(X)

#m = GaussianNB()
m = LogisticRegression()
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


t = time.clock()
m.fit(X_train, Y_train)
print("Fitting time:{}".format(time.clock() - t))
t = time.clock()
predictions = m.predict(X_validation)
print("Prediction time:{}".format(time.clock() - t))
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


im = imageio.imread(sys.argv[1])
V_ = numpy.array(im)
shape = V_.shape
print("Shape of test input:")
print(shape)
V = V_.reshape(1,-1)
V = pca.transform(V)
print(m.predict_proba(V))
#print(predict_proba(V))

'''
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=5, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)    

'''

print("Resource: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
process = psutil.Process(os.getpid())
print("Resource2: {}".format(process.memory_info().rss))

