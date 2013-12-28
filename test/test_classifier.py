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
    
def test_update_weights():
    w0 =  np.array([.2,  0.25, 0.02, 0.04])
    w1 =  np.array([.25, 0.2,  0.02, 0.02])
    dec = np.array([ 1,  0,    0,    1], dtype=np.int)
    exp_w0 = np.array([.2,   0.125, 0.01, 0.04])
    exp_w1 = np.array([.125, 0.2,   0.02, 0.01])
    _sum = exp_w0.sum() + exp_w1.sum()
    exp_w0 /= _sum
    exp_w1 /= _sum
    res_w0, res_w1 = cl.update_weights(w0, w1, 0.5, dec)
    nt.assert_allclose(res_w0, exp_w0)
    nt.assert_allclose(res_w1, exp_w1)
    
def test_apply_transform():
    hashed_table = np.array([0, 1, 2, 0, 1], dtype=np.int)
    w0 = np.array([0.1,  0.05, 0.0, 0.05, 0.2])
    w1 = np.array([0.05, 0.05, 0.1, 0.15, 0.25])
    sum0 = np.array([w0[0] + w0[3], w0[1] + w0[4], w0[2]])
    sum1 = np.array([w1[0] + w1[3], w1[1] + w1[4], w1[2]])
    expected_error = sum0.sum()
    expected_table = np.ones(5)
    _error, _table = cl._apply_transform(hashed_table, w0, w1)
    nt.assert_almost_equal(_error, expected_error)
    nt.assert_allclose(_table, expected_table)  
    del w0, w1, sum0, sum1, expected_error, expected_table, _error, _table
    w0 = np.array([0.13,  0.05, 0.08, 0.15, 0.2])
    w1 = np.array([0.02,  0.05, 0.02, 0.05, 0.25])
    sum0 = np.array([w0[0] + w0[3], w0[1] + w0[4], w0[2]])
    sum1 = np.array([w1[0] + w1[3], w1[1] + w1[4], w1[2]])
    expected_error = sum1[0] + sum0[1] + sum1[2]
    expected_table = np.array([0, 1, 0, 0, 1])
    _error, _table = cl._apply_transform(hashed_table, w0, w1)
    nt.assert_almost_equal(_error, expected_error)
    nt.assert_allclose(_table, expected_table)
    
    
def test_product_hash():
    table = np.array([[0, 0, 0, 1],
                      [0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 1],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1]])
    expected = np.array([8, 6, 5, 11, 15, 3, 12])
    result = cl._product_hash(table)
    nt.assert_allclose(result, expected)

    
    
    

    
    