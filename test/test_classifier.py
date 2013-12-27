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
    
    