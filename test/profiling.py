# -*- coding: utf-8 -*-
import os, sys

thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)
    
import classifier as cl
import numpy as np
import timeit

def run_min():
    ntimes = 1000
    print "Profiling np.min over separate vectors: %d executions." % ntimes
    print timeit.timeit("np.min((w0, w1), axis=0)", 
              setup="import numpy as np; from profiling import error_setup; w0, w1 = error_setup()", number=ntimes)
    print "Profiling np.min over two-column vector: %d executions." % ntimes
    print timeit.timeit("np.min(w, axis=1)", 
              setup="import numpy as np; from profiling import npmin_setup; w = npmin_setup()", number=ntimes)  
    print "Profiling np.min over two-line vector: %d executions." % ntimes
    print timeit.timeit("np.min(w, axis=0)", 
              setup="import numpy as np; from profiling import npmin_setup2; w = npmin_setup2()", number=ntimes)            
              
def npmin_setup():    
    w = np.random.uniform(size=(100000,2))
    _sum = w.sum()
    w /= _sum
    return w

def npmin_setup2():    
    w = np.random.uniform(size=(2, 100000))
    _sum = w.sum()
    w /= _sum
    return w
    
def run_update_weights():
    print timeit.timeit("cl.update_weights(w0, w1, 0.8, dec)", 
                        setup="import classifier as cl;from profiling import update_weights_setup; w0, w1, dec = update_weights_setup()", 
                        number=100)

def run_error():
    print timeit.timeit("cl.error(w0, w1)", 
                        setup="import classifier as cl;from profiling import error_setup; w0, w1 = error_setup()", number=100)
  
def error_setup():    
    w0 = np.random.uniform(size=100000)
    w1 = np.random.uniform(size=100000)
    _sum = w0.sum() + w1.sum()
    w0 /= _sum
    w1 /= _sum
    return w0, w1
    
def update_weights_setup():
    w0 = np.random.uniform(size=100000)
    w1 = np.random.uniform(size=100000)
    dec = cl.make_decision(w0, w1)
    return w0, w1, dec
    
def iteration_setup():
    w0, w1 = error_setup()
    nrows = w0.shape[0]
    ncols = 12
    table = np.random.binomial(1, p=0.3, size=(nrows, ncols))    
    table.reshape((nrows, ncols))
    return table, w0, w1
    
def _single_iteration(table, w0, w1):
    _table = cl.product_hash(table)
    return _table, cl._apply_transform(_table, w0, w1)
     
    
def run_single_iteration():
    print timeit.timeit("_single_iteration", 
                        setup="import classifier as cl;from profiling import iteration_setup, _single_iteration; table, w0, w1 = iteration_setup()", number=200)
    