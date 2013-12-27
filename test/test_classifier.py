# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:14:22 2013

@author: csantos
"""

import os, sys

thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)
    
import classifier as cl
import numpy as np
import numpy.testing as nt

def test_error():
    w0 = np.array([0.15, 0.2, 0.1, 0.03, 0.08])
    w1 = np.array([0.05, 0.0, 0.1, 0.17, 0.12])
    expected_error = w0[[2, 3, 4]].sum() + w1[[0, 1]].sum()
    e0 = cl.error(w0, w1)
    nt.assert_almost_equal(e0, expected_error)
    
def test_update_table():
    w0 =  np.array([.2,  0.25, 0.02, 0.04])
    w1 =  np.array([.25, 0.2,  0.02, 0.02])
    dec = np.array([ 1,  0,    0,    1], dtype=np.int)
    exp_w0 = np.array([.2,   0.125, 0.01, 0.04])
    exp_w1 = np.array([.125, 0.2,   0.02, 0.01])
    #_sum = exp_w0.sum() + exp_w1.sum()
    #exp_w0 /= _sum
    #exp_w1 /= _sum
    res_w0, res_w1 = cl.update_table(w0, w1, 0.5, dec)
    nt.assert_allclose(exp_w0, res_w0)
    nt.assert_allclose(exp_w1, res_w1)
    
    
    