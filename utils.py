# Python 2 compatibility
from __future__ import division
from __future__ import print_function

import os, sys, time, json
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import norm
from copy import deepcopy
from itertools import accumulate
from scipy.interpolate import interp1d

np.random.seed(0x7777777)

color_arr = ('k','r','b','m','y','g','k','r','b','m','y')

""" Check if file exists, if yes, delete it by default input
"""
def checkexists(filename, delete=False):
    if os.path.exists(filename):
        if delete:
            try:
                os.remove(filename)
            except:
                pass
        return True
    return False

""" Central differential of data
"""
def cdiff(data):
    if len(data)==1: return np.array([0])
    d = np.diff(data)
    return np.array([d[0]] + list((d[1:]+d[:-1])/2) + [d[-1]])

""" Central mean of data
"""
def cmean(data):
    if len(data)==1: return np.array([0])
    d = data
    return np.array([d[0]] + list((d[1:]+d[:-1])/2) + [d[-1]])

""" Get Norm2 distance between phi_p and phi_q
"""
def distance(d_pq):
    if len(d_pq.shape)==1:
        return np.abs(d_pq)
    if len(d_pq.shape)>1:
        return np.linalg.norm(d_pq, axis=len(d_pq.shape)-1).reshape(len(d_pq),)

""" Accept-Reject Sampling method
    f(x) is the PDF function
        0 <= f(x) <= 1 should be satisfied
        max(f(x))=1 would be better
"""
def acceptSampling_i(f, xr):
    xmin, xmax = xr
    while True:
        x = np.random.rand()*(xmax-xmin) + xmin
        if np.random.rand() < f(x):
            return x

""" Sampling for given size
"""
def acceptSampling(f, xr, size=(1,)):
    data = np.zeros(size)
    if len(size)>1:
        data = [acceptSampling(f, xr, size=size[1:]) for d in data]
    else:
        data = [acceptSampling_i(f, xr) for d in data]
    return np.array(data)

""" PDF statistics: mean and variance
"""
def PDFstat(xi, yi):
    yi = (cdiff(xi)*yi)
    yi = yi/np.sum(yi)
    mu = np.sum(yi*xi)
    var = np.sum(yi*(xi-mu)**2)
    return mu, var
