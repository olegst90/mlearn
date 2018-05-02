#! /usr/bin/python

import io
import os
import numpy
import resource
import re
import sys
import time
import binascii
import hashlib as hash
from PIL import Image

from sklearn.decomposition import PCA

def memusage():
    print("memusage: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def img_get(path):
    return bytearray(Image.open(path)\
         .convert(mode='L')\
         .resize((640,480),resample=Image.LANCZOS)\
         .tobytes())

filelist=os.listdir('data_str')

X_ = []
Y_ = []

memusage()

for f in filelist:
    im = img_get('data_str/' + f)
    X_.append(numpy.array(list(im)))
    obj_type = f
    Y_.append(f)
    print("Image loaded: " + f)
    memusage()

X = numpy.array(X_)
Y = numpy.array(Y_)
print("Numpy arrays created")
memusage()


pca = PCA(n_components=10, svd_solver='randomized')

X_pca = pca.fit_transform(X)

print("PCA transformation applied")
memusage()


numpy.save("X.opt", X_pca)
numpy.save("Y.opt", Y)

im = img_get(sys.argv[1])
V = numpy.array(list(im))
V = V.reshape(1,-1)
V_pca = pca.transform(V)
numpy.save("V.opt", V_pca)
