# -*- coding: utf-8 -*-
import os, sys

thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)
    
import classifier as cl
import numpy as np
import timeit

def run_update_table():
    print timeit.timeit("cl.update_table(w0, w1, 0.8, dec)", 
                        setup="import classifier as cl;from profiling import update_table_setup; w0, w1, dec = update_table_setup()", 
                        number=100)

def run_error():
    print timeit.timeit("cl.error(w0, w1)", 
                        setup="import classifier as cl;from profiling import error_setup; w0, w1 = error_setup()", number=100)
  
def error_setup():    
    w0 = np.random.uniform(size=10000)
    w1 = np.random.uniform(size=10000)
    _sum = w0.sum() + w1.sum()
    w0 /= _sum
    w1 /= _sum
    return w0, w1
    
def update_table_setup():
    w0 = np.random.uniform(size=10000)
    w1 = np.random.uniform(size=10000)
    dec = cl.make_decision(w0, w1)
    return w0, w1, dec
    
    